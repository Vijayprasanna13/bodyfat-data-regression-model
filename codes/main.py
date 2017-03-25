import time
import sys
import os
import scipy.io as sio
import numpy as np 
import theano
import theano.tensor as T
import lasagne
from cnn_archi import build_cnn
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

X_train = sio.loadmat('data.mat')
X_train = X_train['data']
#y_train = np.arange(26, dtype = np.int32)
#X_train = np.transpose(np.reshape(a,(288000,26)))
X_train = np.reshape(X_train,(100,1,9,1))
X_train = X_train.astype(np.float32)
#X_train = X_train.astype(np.int32)
num_epochs=30

y_train = sio.loadmat('targets.mat')
y_train = y_train['out']
y_train = np.resize(y_train,[100,])
y_train = y_train.astype(np.int32)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3);

epoch_arr = np.arange(num_epochs)+1
val_err_arr = np.empty(num_epochs)
train_err_arr = np.empty(num_epochs)
k = 0;
'''
enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_train = y_train.astype(np.int32)
'''
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
network = build_cnn(input_var)
    
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=1e-4, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
val_fn = theano.function([input_var, target_var], [test_loss])

#y_val = np.reshape(y_val,(20,1))
    # Finally, launch the training loop.
print("Starting training...")
    # We iterate over epochs:
for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
	train_err = 0
        train_batches = 0
        start_time = time.time()
	print(" Epoch :{}".format(epoch+1))
	#count = 0
        for batch in iterate_minibatches(X_train, y_train, 1 , shuffle=True):
            inputs, targets = batch
	    #count = count+1
            train_err += train_fn(inputs, targets)
	    #print(" Iteration:{} ".format(count))
            train_batches += 1
    	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	train_err_arr[k] = train_err/train_batches
	val_err = np.asarray(val_fn(X_val,y_val))
	val_err_arr[k] = val_err
	k=k+1
	print('Validation Loss : {}'.format(val_err))

#......................Plotting errors..................

fig1 = plt.figure()
ax1=fig1.add_subplot(111)

ax1.plot(epoch_arr, val_err_arr, '-', color='b',label='Testing Loss')
ax1.plot(epoch_arr, train_err_arr, '-', color='r',label='Training Loss')

leg = ax1.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
plt.xlabel('Epochs')
plt.ylabel('Error')
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
#ax1.legend([l1,l2],['Validation','Training'])
plt.savefig('1.jpg',bbox_inches='tight')



