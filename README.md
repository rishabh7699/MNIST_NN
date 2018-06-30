# MNIST_NN
This is a project on HandDigit rocognition using regular artificial Neural Networks.

MNIST Dataset is used for traning and testing of network.
This dataset consists of grayscale images of handwritten digits of size 28x28 for both testing and training model.
Tensorflow python library is used to construct and train the model.

Total 5000 images are there in training set which are divided into 50 batches of 100 size each.  
Fpr testing purpose 1000 images are used.
Neural network contains 5 layers in total which include 1 input layer, 1 output layer and 3 hidden layers.
Activation function used in each layer of network is ReLU i.e Rectified Linear Unit.
Optimizer used is Adam Optimizer with learning rate of 0.01.
Dropout is also applied to prevent over fitting of traning set with keep_prob = 0.5

Accuracy of model reached upto 92% on test Dataset and 99.9% on train Dataset.
