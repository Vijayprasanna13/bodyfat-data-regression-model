import matplotlib.pyplot as plt
import numpy as np

X_file_training = np.genfromtxt('housing.csv', delimiter=',', skip_header=1, max_rows=250)
N = np.shape(X_file_training)[0]
print np.shape(X_file_training)[1]
X = np.hstack((
				np.ones(N).reshape(N,1),
				X_file_training[:,1].reshape(N,1),
				X_file_training[:,2].reshape(N,1),
				X_file_training[:,3].reshape(N,1),
				#X_file_training[:,4].reshape(N,1),
				X_file_training[:,5].reshape(N,1),
				X_file_training[:,6].reshape(N,1),
				X_file_training[:,7].reshape(N,1),
				X_file_training[:,8].reshape(N,1),
				X_file_training[:,9].reshape(N,1),
				X_file_training[:,10].reshape(N,1),
				X_file_training[:,11].reshape(N,1),
				X_file_training[:,12].reshape(N,1),
				X_file_training[:,13].reshape(N,1),
				#X_file_training[:,14].reshape(N,1),
			 ))

Y = X_file_training[:,0]
 
for i in range(1,np.shape(X)[1]):
	X[:,i] = (X[:, i]-np.mean(X[:, i]))/np.std(X[:, i])

w = np.zeros(np.shape(X)[1])
eta = 0.29E-3
max_iter = 100

for t in range(0, max_iter):
	gradient = np.zeros(np.shape(X)[1]) 
	for i in range(0, N):
		x_i = X[i, :]
		y_i = Y[i]
		h = np.dot(w,x_i) - y_i
		gradient += 2*x_i*h
	w = w - eta*gradient

X_file_testing = np.genfromtxt('housing.csv', delimiter=',', skip_header=251)
N = np.shape(X_file_testing)[0]
X = np.hstack((
				np.ones(N).reshape(N,1),
				X_file_testing[:,1].reshape(N,1),
				X_file_testing[:,2].reshape(N,1),
				X_file_testing[:,3].reshape(N,1),
				#X_file_testing[:,4].reshape(N,1),
				X_file_testing[:,5].reshape(N,1),
				X_file_testing[:,6].reshape(N,1),
				X_file_testing[:,7].reshape(N,1),
				X_file_testing[:,8].reshape(N,1),
				X_file_testing[:,9].reshape(N,1),
				X_file_testing[:,10].reshape(N,1),
				X_file_testing[:,11].reshape(N,1),
				X_file_testing[:,12].reshape(N,1),
				X_file_testing[:,13].reshape(N,1),
				#X_file_testing[:,14].reshape(N,1),
			 ))
Y = X_file_testing[:,0]
 
for i in range(1,np.shape(X)[1]):
	X[:,i] = (X[:, i]-np.mean(X[:, i]))/np.std(X[:, i])

err = 0

for i in range(0, N):
	print "calculated value :",np.dot(np.transpose(w),X[i,:])," actual value :", Y[i]
	err += abs(Y[i] - np.dot(np.transpose(w),X[i,:]))/Y[i]

print "wieght are : ",w
print "error: ",(err/N)*100,"%"