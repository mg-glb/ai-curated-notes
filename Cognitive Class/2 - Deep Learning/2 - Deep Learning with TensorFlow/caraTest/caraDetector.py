#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

#Load Image
im=misc.imread("cara_principal.jpg").astype(np.float)
#Convert image to gray scale(this elminates another)
grayim=np.dot(im[...,:3],[0.299,0.587, 0.114])
#Convert image into tensor
Image = np.expand_dims(np.expand_dims(grayim,0),-1)
#Create a placeholder with the same size as the input image, and in float format
img = tf.placeholder(tf.float32,[None,773,1200,1])
#Create a placeholder for the output, in our case, we either have Cara or Not Cara
y_ = tf.placeholder(tf.float32,[None,2])

#Define the functions for creating the CNN
def conv2d(X,W):
  return tf.nn.conv2d(input=X,filter=W,strides=[1,1,1,1],padding='SAME')

def MaxPool(X):
  return tf.nn.max_pool(X,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

#Create weights and biases for Convolution
weights = {
  'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
  'W_conv2': tf.Variable(tf.truncated_normal([5,5,32,64]))
}
biases = {
  'b_conv1': tf.Variable(tf.random_normal([32])),
  'b_conv2': tf.Variable(tf.random_normal([64]))
}

#Create the TensorFlow graph
#===========================
#Neurons for the first part
conv1 = conv2d(img,weights['W_conv1'])
relu1 = tf.nn.relu(conv1 + biases['b_conv1'])
maxpool1 = MaxPool(relu1)
#print(relu1.get_shape().as_list())
#print(maxpool1.get_shape().as_list())
#Neurons for the second part
conv2 = conv2d(maxpool1,weights['W_conv2'])
relu2 = tf.nn.relu(conv2 + biases['b_conv2'])
maxpool2 = MaxPool(relu2)
#print(relu2.get_shape().as_list())
#print(maxpool2.get_shape().as_list())
#Neurons for the fully connected layer
layer3_matrix = tf.reshape(maxpool2,[-1,194*300*64])
W_fc1 = tf.Variable(tf.truncated_normal([194*300*64,4],stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1,shape=[4]))
fcl = tf.add(tf.matmul(layer3_matrix, W_fc1),b_fc1)
#Neuron for the final relu
h_fc1 = tf.nn.relu(fcl)
#Neurons for the dropout layer
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
#Neurons for the second fully connected layer
W_fc2 = tf.Variable(tf.truncated_normal([4, 2], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]))
fc=tf.add(tf.matmul(layer_drop, W_fc2), b_fc2)
#Final softmax classifier
y_CNN= tf.nn.softmax(fc)

#Create the Cost function
#========================
cross_entropy_input = tf.reduce_sum(y_ * tf.log(y_CNN),reduction_indices=[1])
cross_entropy = tf.reduce_mean(-cross_entropy_input)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initialize the variables and run the session
init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

#Layer1 = sess.run(maxpool1,feed_dict={img:Image})
#print(Layer1.shape)
Layer2 = sess.run(maxpool2,feed_dict={img:Image})
print(Layer2.shape)
#Layer3 = sess.run(layer3_matrix,feed_dict={img:Image})
#print(Layer3.shape)
#Layer4 = sess.run(fcl,feed_dict={img:Image})
#print(Layer4)
#Layer5 = sess.run(h_fc1,feed_dict={img:Image})
#print(Layer5)
#Layer6 = sess.run(fc,feed_dict={img:Image,keep_prob: 1.0})
#print(Layer6)
#Layer7 = sess.run(y_CNN, feed_dict={img:Image,keep_prob: 1.0})
#print(Layer7)
#Layer8 = sess.run(accuracy,feed_dict={img:Image,keep_prob: 1.0,y_:[[0,1]]})
#print(Layer8)
#Uncomment this so you can print the results of the first layer
'''
vec = np.reshape(Layer1, (387,600,32))
for i in range(32):
  image = vec[:,:,i]
  plt.imshow(image,cmap=plt.get_cmap("gray"))
  plt.savefig("caralayered/{}.png".format(i))
'''
#Uncomment this so you can print the results of the second layer
vec = np.reshape(Layer2,(194,300,64))
for i in range(64):
  image = vec[:,:,i]
  plt.imshow(image,cmap=plt.get_cmap("gray"))
  plt.savefig("caralayered/{}.png".format(i))

sess.close()