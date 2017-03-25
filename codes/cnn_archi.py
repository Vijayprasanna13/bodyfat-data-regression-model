from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

# ........................................Definition of log_softmax().........................................

def log_softmax(x):
   x_diff = x - T.max(x)
   return x_diff - T.log(T.sum(T.exp(x_diff)))

# ..............................................End of log_softmax()..............................................

# .................................Function to build cnn architecture for main.py and test.py...................

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 9, 1),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.

#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=32, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.GlorotUniform())
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
 #123   network = lasagne.layers.Conv2DLayer(
#123            network, num_filters=32, filter_size=(5, 5),
#123            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal())
#123    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    
#    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
#            lasagne.layers.dropout(network,p=0.5), num_filters=50, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeNormal()))
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=32, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify)
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
   

    

#    network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#           num_units=500,
#            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.DenseLayer(
                network,
            num_units=10,
		W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.sigmoid)


    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(network,num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

# ...................................End of Function to build cnn architecture........................................
