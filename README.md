# MNIST_NN
This is a project on Handwritten Digit recognition (MNIST Dataset) using an artificial Neural Network.

The MNIST dataset consists of grayscale images of handwritten digits of size 28x28 pixels.The network have been implemented using tensorflow.

Total 5000 images are there in training set which are divided into 50 batches of 100 size each.
For testing purpose 1000 images have been used.The Neural network contains 5 layers in total which include 1 input layer, 1 output layer and 3 hidden layers. Activation function used in each layer of network is ReLU .Optimizer used is Adam with learning rate of 0.01. Dropout is also applied to prevent over fitting of traning set with keep_prob = 0.5

Accuracy of model reached upto 92% on test Dataset and 99.9% on train Dataset.
