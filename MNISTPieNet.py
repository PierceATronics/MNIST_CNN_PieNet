'''
File Name:	MNISTPieNet.py
Author:		David Pierce Walker-Howell
Date:		06/13/2018
File Description:	The MNISTPieNet class defines a Convolutional Neural Network model to classify digits in the 
					MNIST dataset. The network model is similar in architecture to Yann LeCun's LeNet-5 architecture.
					The MNISTPieNet class includes the model architecure, cost function, training operation, evaluation
					operations and model saving/restoration operations. This network is built in the machine learning
					library tensorflow.
Classes:	MNISTPieNet
'''
import tensorflow as tf
import numpy as np
import os


class MNISTPieNet:
	'''
	MNISTPieNet defines the MNIST CNN architecture and the operations on the 
	'''
	def __init__(self):
		'''
		Class constructor: Initializes class parameters related to the network

		Parameters:
			N/A
		Returns:
			N/A
		'''

		#Height, Width, and num_channels of input data(32x32)
		self.height = 32
		self.width = 32
		self.channels = 1

		#Sizes of filters/Convolutional kernels
		self.conv1_kernel_size = [5, 5]
		self.conv2_kernel_size = [5, 5]
		self.conv3_kernel_size = [5, 5]

		#Kernel size for pooling layers
		self.pool1_kernel_size = [2, 2]
		self.pool2_kernel_size = [2, 2]
		self.pool3_kernel_size = [2, 2]

		self.num_hidden1 = 84 	#Hidden layers number of Neurons in fully connected network
		self.num_outputs = 10	#10 outputs for 10 possible of digits
		self.momentum = 0.9		#Momentum for Nesterov Acclerated Gradient Optimizer
		self.learning_rate = 0.01 


		self.logits = None
		self.softmax = None #Output probability from network for each digit
		self.loss = None
		self.prediction = None
		self.accuracy = None
		self.training_operation = None

		self.default_model_path = "C:\Machine_Learning\ML Projects\MNIST_CNN_PieNet\PieNet_final_model.ckpt"

		#Initialize all instance tensorflow graph attributes
		self.init_all_graphs()

	def init_all_graphs(self):
		'''
		Initialize attribute graphs of the MNIST PieNet object model.

		Parameters:
			N/A
		Returns:
			N/A
		'''
		#Set instance attributes to their graph definitions
		self.logits, self.softmax = self.cnn_model()
		self.loss = self.loss_func()
		self.prediction = self.get_prediction()
		self.accuracy = self.evaluate_accuracy()
		self.training_operation = self.training()

	def cnn_model(self):
		'''
		Definition of graph architecture for the PieNet CNN network.

		Parameters:
			N/A
		Returns:
			logits: The graph definiton for the logits output from the CNN network
			softmax: The softmax output to define the probabilites for the given digits
		'''
		with tf.name_scope("cnn_model"):
			self.X = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='X')
			self.y = tf.placeholder(tf.int32, shape=(None), name='y')

			#He initialization to for a normal weight initialization distribution, also helps avoid vanishing/exploding gradients
			he_initialization = tf.contrib.layers.variance_scaling_initializer()

			#Convolution layer 1: Creates 6 feature maps with output size 28x28
			conv1 = tf.layers.conv2d(inputs=self.X, filters=6, kernel_size=self.conv1_kernel_size, strides=(1, 1), 
				padding="valid", kernel_initializer=he_initialization, activation=tf.nn.relu, name="conv1")

			#Max pooling: downsamples the convolution images to an output size of 14x14
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.pool1_kernel_size, strides=2, 
				padding="valid", name="pool1")

			#Convolution layer 2: Creates 16 feature maps with output size of 10 x 10
			conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=self.conv2_kernel_size, strides=(1, 1), 
				padding="valid", kernel_initializer=he_initialization, activation=tf.nn.relu, name="conv2")

			#Max pooling: downsamples the convolution images to an output size of 14x14
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.pool2_kernel_size, strides=2, 
				padding="valid", name="pool2")

			#Convolution layer 3: Creates 120 feautre maps with output size of 1 x 1
			conv3 = tf.layers.conv2d(inputs=pool2, filters=120, kernel_size=self.conv3_kernel_size, strides=(1, 1), 
				padding="valid", kernel_initializer=he_initialization, activation=tf.nn.relu, name="conv3")

			
			#Flatten out the convolution data to be passed through FNN
			flattend_conv3 = tf.reshape(conv3, [-1, 1 * 1 * 120])

			#Fully connected layers
			hidden1 = tf.layers.dense(flattend_conv3, self.num_hidden1, activation=tf.nn.relu, 
				kernel_initializer=he_initialization, name="hidden1")

			#Ouptput layer: Should output 10 probabilites to predict correct digit
			logits = tf.layers.dense(hidden1, self.num_outputs, activation=None, kernel_initializer=he_initialization,
				name="logits")

			#Compute Softmax Regression 
			softmax = tf.nn.softmax(logits, axis=None, name="softmax")
		return logits, softmax

	def loss_func(self):
		'''
		Defines the cost function for the multiclass output. A softmax function with a cross
		entropy cost function is used to measure loss to train network.

		Parameters:
			N/A
		Returns:
			loss: Definition of the loss graph
		'''
		with tf.name_scope("loss"):
			#Softmax regression is used to support multiple class output. The function uses a "Normalized Exponential" 
			#to obtain the probability of each class. Cross entropy
			#self.x_entropy is a tensor with the same shape as labels containing the cross entropy loss 
			x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="x_entropy")

			#Take the mean across the entire mini-batch
			loss = tf.reduce_mean(x_entropy, name="loss")
		return loss


	def evaluate_accuracy(self):
		'''
		Evaluate the accuracy of the CNN. Uses the softmax output and target values to compute accuracy

		Parameters:
			N/A
		Returns:
			accuracy: Definition of the accuracy graph
		'''
		with tf.name_scope("evaluate_training_accuracy"):
			#Compute if the largest probability for a digit from the softmax corresponds to the correct label
			correctly_labeled = tf.nn.in_top_k(self.softmax, self.y, 1)
			accuracy = tf.reduce_mean(tf.cast(correctly_labeled, tf.float32))
		return accuracy
	
	def get_prediction(self):
		'''
		Using the softmax probabilites from the net work, return the predicted digit based on the highest probability

		Parameters:
			N/A
		Returns:
			predicitions: The predicition (an integer between [0-9]) for images ran through the network
		'''
		with tf.name_scope("predicition"):
			#retrieve the top value(probability) from the softmax predicition and index of the top probability
			values, indicies = tf.nn.top_k(self.softmax, k=1)
		
			#since the index number corresponds directly to the digit(ex: index 5 is probability of digit being 5), the
			#prediction is the same as the top probability index
			predictions = indicies
			return predictions

	def training(self):
		'''
		Defines the training process for the CNN. First the logits and softmax graphs are initialized from
		the CNN model. Next the loss, Nesterov Acclerated Gradient optimizer, and training operation
		graphs are initialized. 

		Parameters: 
			N/A
		Returns:
			training_operation: The definition of the graph workflow for training the CNN from scratch
		'''
		with tf.name_scope("train"):
			#Initialize the CNN model graph(can return logits and softmax probabilites)
			#self.logits, self.softmax = self.cnn_model()

			#Initialize the cost(loss) function needed to train the network
			#self.loss_func = self.loss()

			#Define Nesterov Acclerated Gradient Optimizer
			NAG_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, 
				use_nesterov=True, name="NAG_optimizer")

			#Train the model by minimizing the loss using NAG Optimizer
			training_operation = NAG_optimizer.minimize(self.loss, name="training_operation")

		return training_operation

	def save_final_model(self, tf_session, path=None):
		'''
		Save the final model the as a tensorflow binary file to specified path

		Parameters:
			tf_session: The tensorflow session to be saved
			path: Default will be 'self.default_model_path'. Else specify own path to save model binaries

		Returns:
			N/A
		'''
		if path is None:
			path = self.default_model_path

		#Model saver node
		self.saver = tf.train.Saver()

		try:
			self.final_model_path = self.saver.save(tf_session, path)
		except IOError:
			print("Either file path ("+ path +") or tf.Session() (" + tf_session +") does not exist")

	def restore_final_model(self, tf_session, path=None):
		'''
		Restore a saved final model.
		
		Parameters:
			tf_session: The tensorflow session to be saved
			path: Default will be 'self.default_model_path'. Else specify own path to save model binaries

		Returns:
			N/A

		'''
		if path is None:
			path = self.default_model_path

		#Model saver node
		self.saver = tf.train.Saver()

		try:
			self.saver.restore(tf_session, path)
		except IOError:
			print("Either file path ("+ path +") or tf.Session() (" + tf_session +") does not exist")


	def tensorboard_summaries(self):
		'''
		Defines the nodes for tensorboard summaries. The two summary nodes that post to tensorboard are
		cross entropy loss and accuracy.

		Parameters:
			N/A
		Returns:
			N/A
		'''
		#Tensorboard summaries
		self.x_entropy_loss_summary = tf.summary.scalar('X_Entropy_Loss', self.loss)
		self.accuracy_summary = tf.summary.scalar('Accuracy', self.accuracy)
		self.all_summaries_merged = tf.summary.merge_all()

