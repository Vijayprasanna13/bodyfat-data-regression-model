import matplotlib.pyplot as plt
import numpy as np
import random

asdas
asdasd
#Set the learning rate and No of epochs
eta = 0.3E-3
max_hidden_nodes = 7
max_epochs = 40
filename = 'bodyfat.csv'

fid = open(filename, "r")
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("shuffled_bodyfat.csv", "w")
fid.writelines(li)
fid.close()

filename = "shuffled_bodyfat.csv"
print "training ..."
#Split the total number of training sets into two halves. One for training and the other for testing
max_training = np.shape(np.genfromtxt(filename, delimiter=','))[0]/2

#Load the training data set
X_file_training = np.genfromtxt(filename, delimiter=',', max_rows=max_training)
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


sse = []
sse_testing_i = []
epoch_index = []

#Run for max_epochs number of epochs and update the wieght by calculating the gradient. 
#Record the cost function output for error for each epoch
for t in range(0, max_epochs):
	gradient = np.zeros(np.shape(X)[1]) 
	for i in range(0, N):
		x_i = X[i, :]
		y_i = Y[i]
		
		#sigmoidal
		#h = 1/(1+np.exp(-(np.dot(np.transpose(w),x_i))))
		#gradient += 2*(1-h)*(h)*(h-y_i)
		#linear
	 	h = np.dot(np.transpose(w),x_i) - y_i
		gradient += 2*x_i*h
	w = w - eta*gradient
	sse_i = 0
	for i in range(0, N):
		x_i = X[i, :]
		y_i = Y[i]
		sse_i += pow(( np.dot(np.transpose(w),x_i) - y_i),2)
	if t%10 == 0:
		err = 0
		for k in range(0, N):
			err += abs(Y[k] - np.dot(np.transpose(w),X[k,:]))/Y[k]
		epoch_index.append(t)
		sse_testing_i.append(err)
	sse.append(sse_i)

print "Testing ... "
#Load the testing data set
X_file_testing = np.genfromtxt(filename, delimiter=',', skip_header=max_training+1)
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

print np.shape(w), np.shape(X)
#Find the accuracy and error from target and calculated output
for i in range(0, N):
	err += abs(Y[i] - np.dot(np.transpose(w),X[i,:]))/Y[i]
print "Wieghts : ",w
print "Error: ",(err/N)*100,"%"
epochs = [epoch_i+1 for epoch_i in range(0,max_epochs)]


#Plot the SSE vs Epochs
training_curve, = plt.plot(epochs, sse, label="training set")
#plot the SSE vs epochs for every 10 epochs
testing_curve, = plt.plot(epoch_index, sse_testing_i, label="testing set")

plt.legend([training_curve, testing_curve],["training","testing"])
plt.xlabel('Epochs')
plt.ylabel('SSE')
plt.title('BodyFat ANN Error:'+str((err/N)*100)+'%')
plt.savefig('Eta:'+str(eta)+'.png')

print "Completed. SSE vs Epochs plot for the given Eta is saved in the project directory.\n"

#remove this comment to get predicted values and calculated values
plt.cla()

actual_data_i = []
predicted_i = []
for i in range(0, N):
	predicted_i.append(np.dot(np.transpose(w),X[i,:]))
	actual_data_i.append(Y[i])
n = len(predicted_i)
predicted = plt.scatter([i for i in range(0,n)],predicted_i,c="r",alpha=0.5)
actual = plt.scatter([i for i in range(0,n)],actual_data_i,c="b",alpha=0.5)
plt.legend([predicted, actual],["predicted", "actual"])
plt.xlabel('data row no.')
plt.ylabel('Target')
plt.title("Deviation")
plt.savefig("target deviation for Eta:"+str(eta)+".png")
