#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

#The goal of this tutorial is to learn how to read sample images and use them as input for CNNs.
#A=READ THE DATA
#===============
#1 - Read the image as float data type
im=misc.imread("lena.png").astype(np.float)

#2 - Convert image to gray scale
grayim=np.dot(im[...,:3],[0.299,0.587, 0.114])

'''
#Use matplotlib to create the float and grayscale images
plt.subplot(1,2,1)
plt.imshow(im)
plt.xlabel(" Float Image ")

plt.subplot(1,2,2)
plt.imshow(grayim, cmap=plt.get_cmap("gray"))
plt.xlabel(" Gray Scale Image ")

plt.savefig("lenaFiltered.png")
'''
#3 - Extend the dimensions of the grayscale image
Image = np.expand_dims(np.expand_dims(grayim,0),-1)
#print(Image.shape)

#4 - Create a placeholder with the same size as the input image, and in float format
img = tf.placeholder(tf.float32,[None,512,512,1])
#print(img.get_shape().as_list())

#5 - Create a variable for weight matrix and print the shape
#    the shape of weight matrix is of the form: [height,width,input, output].
#    Create weight matrix of size 5x5 and keep the number of inputs and outputs
#    to just one. So, the shape is of the form [5,5,1,1]
shape=[5,5,1,1]
weights = tf.Variable(tf.truncated_normal(shape,stddev=0.05))
#print(weights.get_shape().as_list())

#6 - Create two convolution graphs in TensorFlow
#    Using tf.nn.conv2d, create two graphs, one using 'same' and the other
#    using 'valid' padding.
ConOut = tf.nn.conv2d(input=img,
                      filter=weights,
                      strides=[1,1,1,1],
                      padding='SAME')

ConOut2 = tf.nn.conv2d(input=img,
                      filter=weights,
                      strides=[1,1,1,1],
                      padding='VALID')

'''
#Run the sessions to get the results for two convolution operations
result = sess.run(ConOut,feed_dict={img:Image})
result2 = sess.run(ConOut2,feed_dict={img:Image})

#Display the output images
#    The result of convolution with 'same' padding is of the form [1,512,512,1]
#    and for 'valid' padding image is of the shape [1,508,508,1]. To display 
#    the images, our job is to reshape the dimensions in the form (512,512) and
#    (508, 508 respectively).
#a - For the result with 'SAME' Padding.
#    Reduce the dimension
vec = np.reshape(result,(1,-1))
#    Reshape the image
image =np.reshape(vec,(512,512))
#b - For the result with 'VALID' Padding.
#    Reduce the dimension
vec2 = np.reshape(result2, (1, -1))
#    Reshape the image
image2= np.reshape(vec2,(508,508))
#c - Plot them in pyplot
plt.subplot(1, 2, 1)
plt.imshow(image,cmap=plt.get_cmap("gray"))
plt.xlabel(" SAME Padding ")

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap=plt.get_cmap("gray"))
plt.xlabel(" VALID Padding ")

plt.savefig("lenaConvolved.png")
'''

#B=CREATE THE CNN
#8 - Create the functions that will be part of the net. First we will form part
#of the convolution layer.
def conv2d(X,W):
  return tf.nn.conv2d(input=X,filter=W,strides=[1,1,1,1],padding='SAME')
#Then we create the maxpooling function
def MaxPool(X):
  return tf.nn.max_pool(X,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

#9 - Create weights and biases for Convolution
#The weights are of the shape [height, width, input, output]. In our case, the
#images will be of size 5x5 with 1 input and 32 outputs.
weights = {
  'W_conv1': tf.Variable(tf.random_normal([5,5,1,32]))
}
biases = {
  'b_conv1': tf.Variable(tf.random_normal([32]))
}

#10 - Define a TensorFlow graph for Relu, Convolution and Maxpooling.
#First send img and W_conv1 as inputs for conv2d. To that result, add b_conv1, 
#and send it to the relu. Finally send that result to MaxPool.
conv = conv2d(img,weights['W_conv1'])
conv1 = tf.nn.relu(conv + biases['b_conv1'])
Mxpool = MaxPool(conv1)

#print(conv1.get_shape().as_list())
#print(Mxpool.get_shape().as_list())

#11 - Initialize the variables and run the session
init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

#12 - Create the first layer of the CNN
Layer1 = sess.run(Mxpool,feed_dict={img:Image})


vec = np.reshape(Layer1, (256,256,32))
for i in range(32):
  image = vec[:,:,i]
  plt.imshow(image,cmap=plt.get_cmap("gray"))
  plt.xlabel(i,fontsize=20,color='red')
  plt.savefig("lenalayered/{}.png".format(i))

sess.close()