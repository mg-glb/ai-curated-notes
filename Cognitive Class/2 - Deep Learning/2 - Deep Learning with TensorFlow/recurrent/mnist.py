#TensorFlow comes with helper functions to download and process MNIST.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#The function input_data.read_data_sets() loads the entire dataset and returns an object tensorflow.contrib.learn.python.learn.datasets.mnist.DataSets.
#The argument (one_hot=False) creates the label arrays as 10-dimensional binary vectors (only zeros and ones), in which the index cell for the number one, is the class label.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_imgs = mnist.train.images
train_labels = mnist.train.labels
test_imgs = mnist.test.images
test_labels = mnist.test.labels

#The train images shape is 55000 inputs and 784 pixels by images.
ntrain = train_imgs.shape[0]
#The test images shape is 10000 inputs and 784 pixels by images.
ntest = test_imgs.shape[0]
#The train class shape is 55000 inputs and 10 classes.
dim = train_imgs.shape[1]
#The test class shape is 10000 inputs and 10 classes.
nclasses = train_labels.shape[1]

#We will treat the MNIST image as 28 sequences of a vector of length 28.
#Our simple RNN will consist of:
#1-one input layer which converts a 28x28 dimensional input to an 128 dimensional hidden layer.
#2-One intermediate recurrent neural network (LSTM).
#3-One output layer which converts an 128 dimensional output of the LSTM to 10 dimensional output indicating a class label.
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

#Construct the RNN
#The input should be a Tensor of Shape [batch_size,time_steps,input_dimension] but in our case it will be [batch_size,28,28].
#Current data input shape: (batch_size, n_steps, n_input) [100x28x28]
x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x")
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")
#Let's create the weights and biases for the read out layer.
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Now let's define the LSTM cell with TensorFlow.
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
#dynamic_rnn creates a recurrent neural network specified from lstm_cell
outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

#The output of the rnn would be a [100x28x128] matrix. We use the linear activation to map it to a[batch_size,10]
output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
pred = tf.matmul(output, weights['out']) + biases['out']

#Now, we define the cost function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here we define the accuracy and evaluation methods to be used in the learning process:
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Just recall that we will treat the MNIST image as 28 sequences of a vector of length 28.
#So, let's run the Session.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    #Keep training until reach max iterations
    while step * batch_size < training_iters:
        #We will read a batch of 100 images [100x784] as batch_x
        #batch_y is a matrix of [100x10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #We consider each row of the image as one sequence
        #Reshape data to get 28 seq of 28 elements, so that, batch_x is [100x784]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        #Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #Display every ten iterations
        if step % display_step == 0:
            #Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x:batch_x,y:batch_y})
            #Calculate batch loss
            loss = sess.run(cost, feed_dict={x:batch_x, y: batch_y})
            print("Iter %s, Minibatch Loss= %s, Training Accuracy= %s" % (str(step*batch_size),"{:.6f}".format(loss),"{:.5f}".format(acc)))
        step += 1
    print("Optimization finished!")
    #Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy: %s" % sess.run(accuracy,feed_dict={x: test_data, y:test_label}))