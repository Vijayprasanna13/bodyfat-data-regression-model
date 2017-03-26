import matplotlib.pyplot as plt
import numpy as np

#Split the total number of training sets into two halves. One for training and the other for testing
max_training = np.shape(np.genfromtxt('bodyfat.csv', delimiter=','))[0]/2

#Load the training data set
X_file_training = np.genfromtxt('bodyfat.csv', delimiter=',', max_rows=max_training)
N = np.shape(X_file_training)[0]

#Extract all the available column values from the given training set  
features = []
for i in range(0,np.shape(X_file_training)[1]):
	if i == 0:
		features.append(np.ones(N).reshape(N,1))
	else:			
		features.append(X_file_training[:,i].reshape(N,1))

#Create the input np array
X = np.hstack((features))
Y = X_file_training[:,0]

#Scale the input vectors using (col.values - mean)/std.dev
for i in range(1,np.shape(X)[1]):
	X[:,i] = (X[:, i]-np.mean(X[:, i]))/np.std(X[:, i])
w = np.zeros(np.shape(X)[1])

#Set the learning rate and No of epochs
eta = 0.2E-3
max_epochs = 100
sse = []

#Run for max_epochs number of epochs and update the wieght by calculating the gradient. 
#Record the cost function output for error for each epoch
for t in range(0, max_epochs):
	gradient = np.zeros(np.shape(X)[1]) 
	for i in range(0, N):
		x_i = X[i, :]
		y_i = Y[i]
		h = np.dot(w,x_i) - y_i
		gradient += 2*x_i*h
	w = w - eta*gradient
	sse_i = 0
	for i in range(0, N):
		x_i = X[i, :]
		y_i = Y[i]
		sse_i += pow(( np.dot(np.transpose(w),x_i) - y_i),2)  
	sse.append(sse_i)

#Load the testing data set
X_file_testing = np.genfromtxt('bodyfat.csv', delimiter=',', skip_header=max_training+1)
N = np.shape(X_file_testing)[0]

#Load all available features
features = []
for i in range(0,np.shape(X_file_testing)[1]):
	if i == 0:
		features.append(np.ones(N).reshape(N,1))
	else:			
		features.append(X_file_testing[:,i].reshape(N,1))
X = np.hstack((features))
Y = X_file_testing[:,0]

#Scale the testing input
for i in range(1,np.shape(X)[1]):
	X[:,i] = (X[:, i]-np.mean(X[:, i]))/np.std(X[:, i])
err = 0

#Find the accuracy and error from target and calculated output
for i in range(0, N):
	err += abs(Y[i] - np.dot(np.transpose(w),X[i,:]))/Y[i]
print "Wieghts : ",w
print "Accuracy: ",(1-(err/N))*100,"%"
epochs = [epoch_i+1 for epoch_i in range(0,max_epochs)]

#Plot the SSE vs Epochs
plt.plot(epochs, sse)
plt.xlabel('Epochs')
plt.ylabel('SSE')
plt.title('BodyFat ANN Accuracy:'+str((1-(err/N))*100)+'%')
plt.savefig('Eta:'+str(eta)+'.png')
