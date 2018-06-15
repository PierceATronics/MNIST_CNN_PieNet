# MNIST_CNN_PieNet #

## Description ##
PieNet is a Convolutional Neural Network (CNN) model built with TensorFlow for classifying digits from the popular MNIST dataset. This CNN architecutre is based off of the LeNet-5 architecture developed by Yann LeCun (see http://yann.lecun.com/). 

## Prerequisites ##
* python3
* Tensorflow
* Numpy
  
## Architecture ##
Layer         | Num_Maps      | Size          | Kernel Size   | Stride        | Activation
------------- | ------------- | ------------- | ------------- | ------------- | -------------
Input         | 1             |32x32          | -             | -             | -
Convolution 1 | 6             |28x28          | 5x5           | 1             | ReLu
Max Pooling   | 6             |14x14          | 2x2           | 2             | -
Convolution 2 | 16            |10x10          | 5x5           | 1             | ReLu
Max Pooling   | 16            |5x5            | 2x2           | 2             | -
Convolution 3 | 120           |1x1            | 5x5           | 1             | ReLu
Fully Conn.   | -             |84             | -             | -             | ReLu
Fully Conn.   | -             |10             | -             | -             | Softmax

## Training ##
Although I have included a pretrained model, it is simple to perform accuracte training in a short amound of time (<1hr) on a standard CPU. Currently, the default training uses TensorFlow's MNIST example dataset which contains 55,000 training images of digits. With Mini-batch training, with batchs sizes of 100 and using a Nesterov Acclerated Gradient Optimizer, the network achieved ~97% accuracy on the validation set over only 40 epochs. Here is how to perform the training.(Note: You can change the batch size and epochs in main.py).

`python main.py train`

The Accuracy data and Cross Entropy Cost Function data will automatically be saved to be view through TensorBoard. The default location of the **tf_logs** directory is `[src code directory]/tf_logs`. To view the graph architecture and data graphs on TensorBoard, run:

`tensorboard --logdir /tf_logs`

![picture alt](tensorboard_image.jpg)

The model parameters and graph structure will also be saved and able to be restored for both inferencing and/or re-training. By default the final model save is located in directory `C:\Machine_Learning\ML Projects\MNIST_CNN_PieNet\PieNet_final_model.ckpt`. This can be changed by editing the MNISTPieNet class attribute `self.default_model_path`. Note: Documentation in the code is provided to easily save and restore models.

## Inferencing ##
After having a pretrained model(I include an already trained model), it is easily to do a simpe inference on a number of MNIST images provide in the Tensorflow example set. Simply run:

`python main.py inference [number_of_images_to_inference]`

Example: `python main.py inference 100`. This inferences 100 MNIST images.
