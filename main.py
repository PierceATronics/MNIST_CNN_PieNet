import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from MNISTPieNet import MNISTPieNet 
import sys
from datetime import datetime

def preprocess_images(images):
	'''
	Preprocess the MNIST images. The current image data past are of shape (55000,784), where 55000 is 
	the number of image and 784 is a 1D vecotr of image data. However the CNN model used takes a 4D input
	tensor with shape (batch_size, image_height, image_width, image_channel). For this CNN model, 
	height x width x channels = 32x32x1. Once the MNIST images are  converted to a two a size of 28x28x1, they
	are then zero padded to be size 32x32x1.

	Parameters:
		images:
	'''
	#If input shate is of shape (batch_size, num_pixels) 
	if(len(images.shape) == 2):
		images = np.reshape(images, [-1, 28, 28, 1])	 

	#If a single images is passed, it should be a shape of (num_pixels) with size 784 pixels
	elif(len(images.shape) == 1):
		images = [np.reshape(images, [28, 28, 1])] 

	else:
		pass
	#pad the images with zeros
	padded_images = np.pad(images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	return padded_images

def get_mini_batch(X, y, current_batch_num, batch_numbers):
	'''
	Return the mini batch for input and output data given the current iteration.

	Parameters:
		X: The input data to the CNN. NOTE: pass the whole data set
		y: The output(target data) corresponding to the input data. NOTE: pass the whole target data set
		current_batch_num: The current iteration in mini-batch gradient descent used to select the next batch
		batch_numbers: The number of batch to partition data into
	Returns:
		batch_X: The mini-batch for the input data
		batch_y: The mini-batch for the output(target) data
	'''
	batch_X = np.split(X, batch_numbers, axis=0)[current_batch_num]
	batch_y = np.split(y, batch_numbers, axis=0)[current_batch_num]
	batch_y = np.reshape(batch_y, -1) #Make a 1D vector 
	return batch_X, batch_y

def train_model(n_epochs=40, batch_size=100):
		'''
		Train MNIST PieNet Convolutional Neural Network with TensorFlows mnist example dataset.

		Parameters:
			n_epochs: Number of epochs(times) to run through the whole dataset during training
			batch_size: Size of mini-batchs for mini-batch gradient descent
		Returns:
			N/A
		'''
		#Downloaded and preprocess MNIST image data
		mnist = input_data.read_data_sets("/tmp/data")

		#Preprocess the images to have zero padding and be dimension 32x32
		training_images = preprocess_images(mnist.train.images)
		validation_images = preprocess_images(mnist.validation.images)

		#Get the target labels
		training_labels = mnist.train.labels 
		validation_labels = mnist.validation.labels

		#Initialize an instance of PieNet and define the training graph model
		MNIST_cnn_model = MNISTPieNet()
		init = tf.global_variables_initializer()

		#Initialize summary nodes for tensorboard
		MNIST_cnn_model.tensorboard_summaries()
		file_writer = tf.summary.FileWriter(tf_board_log_directory(), tf.get_default_graph()) #File writer to write to tensorboard

		#Number of batches
		num_batches = training_images.shape[0] // batch_size
		num_batches = 10
		with tf.Session() as sess:
			#Initialize global variables
			init.run()

			for epoch in range(n_epochs):
				for iteration in range(num_batches):
					X_batch, y_batch = get_mini_batch(training_images, training_labels, iteration, num_batches)
					
					#Write data summaries to tensorboard
					if iteration % 10 == 0:
						step = epoch * num_batches + iteration

						#Plot validation summaries to tensorboard
						summary_str = MNIST_cnn_model.all_summaries_merged.eval(feed_dict={MNIST_cnn_model.X:validation_images, 
							MNIST_cnn_model.y:validation_labels})
						file_writer.add_summary(summary_str, step)

					sess.run(MNIST_cnn_model.training_operation, feed_dict={MNIST_cnn_model.X:X_batch, MNIST_cnn_model.y:y_batch})

				#Training accuracy
				acc_train = MNIST_cnn_model.accuracy.eval(feed_dict={MNIST_cnn_model.X:X_batch, MNIST_cnn_model.y:y_batch})
				
				#Validation set accuracy
				acc_val = MNIST_cnn_model.accuracy.eval(feed_dict={MNIST_cnn_model.X:validation_images, MNIST_cnn_model.y:validation_labels})
				print(epoch, "Training Accuracy:", acc_train, "| Validation Accuracy:", acc_val)

			#Save final model
			MNIST_cnn_model.save_final_model(sess)
			file_writer.close()
		print("Model Training Complete")
		return
def classify_digit(images, labels=None):
	'''
	Use a pretrained model of MNIST PieNet to inference/classifiy mnist digits. The predicitions and 
	accuracy of predicitions(if applicable) will be displayed. 	

	Parameters:
		images: MNIST data set images. images must have shape (num_images, 784)
		labels: Default None. The labels to show if digits are classified by model correctly.

	Returns:
		N/A
	'''
	#Initialize model
	MNIST_cnn_model = MNISTPieNet()

	#Preprocess image
	images = preprocess_images(images)
	
	with tf.Session() as sess:
		MNIST_cnn_model.restore_final_model(sess)

		predictions = np.reshape(MNIST_cnn_model.prediction.eval(feed_dict={MNIST_cnn_model.X:images}), -1)
		for index, pred in enumerate(predictions):
			
			if(labels.all() != None):
				print("Predicted Digit: " + str(pred) +" | " + "Actual Digit: " + str(labels[index]))
			else:
				print("Predicted Digit: " + str(pred))
		if(labels.all() != None):
			accuracy = MNIST_cnn_model.accuracy.eval(feed_dict={MNIST_cnn_model.X:images, MNIST_cnn_model.y:labels})
			print("Percentage of digits correctly classified: " + str(accuracy * 100) + "%")

def tf_board_log_directory():
	'''
	Create a tensorboard summary file to be posted in the tensorboard log directy.
	Each summary file needs to be unique, hence data and time is used in summary file
	name.

	Parameters:
		N/A
	Returns:
		log_dir: The log directy with summary file used to save data to tensorboard.
	'''
	now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
	root_logdir = "tf_logs"
	log_dir = "{}/run-{}".format(root_logdir, now)
	return(log_dir)


def main():
	'''
	Main controls how the MNIST PieNet Convolutional Neural Network will be used. There are two modes:
	The first mode is the training mode to train the network on tensorflows MNIST dataset provided.
	The second mode is the inferencing mode. Inferencing mode loads up the parameters of an already trained 
	network and inferences the images passed.

	For training run: python main.py train
	For inferencing run: python main.py inference <num_of_images_to_inference>

	Parameters:
		N/A
	Returns:
		N/A
	'''
	#specifiy to use training mode
	if(sys.argv[1] == 'train'):
		train_model()

	#specify to use inferencing mode
	elif(sys.argv[1] == 'inference'):
		#Number of images to inference from the tensorflow mnist validation set
		num_images = int(sys.argv[2])

		mnist = input_data.read_data_sets("/tmp/data")
		classify_digit(mnist.validation.images[:num_images], mnist.validation.labels[:num_images])

	#Help if invalid command is given to script
	elif(sys.argv[1] == '--help'):
		print("For inferencing mode: python main.py inference <num_of_images>")
		print("For training mode: python main.py train")
	else:
		print("Invalid Arguments passed! Please try 'python main.py --help' for details")

if __name__ == "__main__":
	#Run main script
	main()
	
