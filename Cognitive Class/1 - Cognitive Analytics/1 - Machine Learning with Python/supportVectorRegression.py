#This small demo is about svm's. However you should take a more indepth course about this method.
from sklearn.svm import SVR
import numpy as np
#Import the libraries to draw the plots.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

#Set the random seed to ensure the same results with the random.seed function of np with parameter 5.
np.random.seed(5)

#Now we will generate 30 different values using the random.rand function of np with parameters 30, 1.
#This will be multiplied by 10, sort it using the np.sort function with axis=0 and store it in X.
X = np.sort(10 * np.random.rand(30, 1), axis=0)
#Now we will take the sin of each using the sin function of np (sinusoidal function).
#Then use the ravel function to format it correctly. This will be stored as y.
y = np.sin(X).ravel()

#We instantiate the Suppor Vector Machine
#Now we will be looking at 3 different kernels that SVR uses: rbf, linear, and sigmoid.
#First we will show you how to declare a rbf kernel.
#Then we will see let finish up for linear and sigmoid!
#Here we define a variable called svr_rbf using the SVR declaration with parameters kernel='rbf' and C=1e3.
svr_rbf = SVR(kernel='rbf', C=1e3)
svr_linear = SVR(kernel='linear', C=1e3)
svr_sigmoid = SVR(kernel='sigmoid', C=1e3)
#Train the models
svr_rbf.fit(X,y)
svr_linear.fit(X,y)
svr_sigmoid.fit(X,y)
#Use the models to predict some values.
y_pred_rbf = svr_rbf.predict(X)
y_pred_linear = svr_linear.predict(X)
y_pred_sigmoid = svr_sigmoid.predict(X)

#Now draw the model
plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_pred_rbf, c='g', label='RBF model')
plt.plot(X, y_pred_linear, c='r', label='Linear model')
plt.plot(X, y_pred_sigmoid, c='b', label='Sigmoid model')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('svm.png')

#Take note that I need to research about kernels for models.