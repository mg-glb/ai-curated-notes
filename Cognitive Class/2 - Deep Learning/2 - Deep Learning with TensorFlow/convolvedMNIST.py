#The purpose of this model is to improve from the one we defined in simpleModel.py
#This CNN will have the following layers:
#1-(Input): [batch_size,28,28,1]>>Apply filter of [5x5]
#2-(Convolutional layer 1)>>[batch_size,28,28,32]
#3-(ReLU 1)>>[batch_size,28,28,32]
#4-(Max pooling 1)>>[batch_size,14,14,32]
#5-(Convolutional Layer 2)>>[batch_size,14,14,64]
#6-(ReLU 2)>>[batch_size,14,14,64]
#7-(Max pooling 2)>>[batch_size,7,7,64]
#8-(Fully Connected Layer 1)>>[batch_sizex1024]
#9-(ReLU 3)>>[batch_sizex1024]
#10(Drop out)>>[batch_sizex1024]
#11(Fully Connected Layer 2)>>[1x10](Output)
#Let's get the inputs from simpleModel.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Create the initial parameters. We save them, so that we can use them accross the script.
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image (in this case 784)
class_output = 10 # number of possible classifications for the problem (in this case the range from 0 to 9)
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Here comes the first difference with simpleModel.py: we reshape the x placeholder from a [batch_size,784] matrix, to a
# [batch_size,28,28,1] tensor. Some notes here:
# -Since the batch size is not defined in the placeholder, but in the feed_dict, we can put any value here, like -1.
# -The area of the image is 28x28=784 pixels^2. If we give a 28x28 parameter, we can "square the array"
# -The final parameter is the image channel. Since we have information in black and white only (no grayscale or colors),
# we only need one channel.
# Check tensorflow's reference on the reshape function.
x_image = tf.reshape(x, [-1,28,28,1])


#CREATE THE CONVOLUTIONAL LAYERS
#===============================
#Layer 1: Convolutional Layer: [batch_size,28,28,1]>>[batch_size,28,28,32]
#We define a kernel of size 5x5. We need 32 different feature maps (what implies 32 neurons in this layer).
# 1-The number of input channels is 1 (greyscale).
# 2-The physical dimensions of the image are 28x28. Since the filter has 5x5 by doing (32-5+1)x(32-5+1)x32, the output
#   of our convolution will be 28x28x32.
# 3-In this step, we create a filter / kernel tensor of shape:
#   [filter_height, filter_width, in_channels, out_channels]=[5,5,1,32]
#This is the weight tensor of the neuron. We will convolve this filter with the image.
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
#Since we have 32 neurons, we will have 32 biases.
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

#We convolve x_image and W_conv1 and then add b_conv1 on top. Let's look in detail:
#INPUTS
#x_shape: tensor of shape [batch, in_height, in_width, in_channels]. x of shape [batch_size,28 ,28, 1]
#W: a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]. W is of size [5, 5, 1, 32]
#stride: A 1-D tensor which is [1, 1, 1, 1]. The stride is the number of positions we shift the filter through the image
#        at each convolution step.
#        -As the first and last dimensions are related to batch and channels, we set the stride to 1.
#        -But for second and third dimensions(height and width), we coould set other values, e.g. [1, 2, 2, 1]
#PROCESS:
# 1-Change the filter to a 2-D matrix with shape [5*5*1,32].
# 2-Extracts image patches from the input tensor to form a virtual tensor of shape [batch, 28, 28, 5*5*1].
# 3-For each batch, right-multiplies the filter matrix and the image vector.
#OUTPUT:
#A Tensor (a 2-D convolution) of size <tf.Tensor 'add_7:0' shape=(?, 28, 28, 32).
# Notice: the output of the first convolution layer is 32 [28x28] images.
# Here 32 is considered as volume/depth of the output image.
convolve1= tf.add(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME'),b_conv1)

#Layer 2: Relu 1 [batch_size,28,28,32]>>[batch_size,28,28,32]
#In this step, we just go through all outputs convolution layer, convolve1.
#Wherever a negative number occurs, we swap it out for a 0. It is called ReLU activation Function.
h_conv1 = tf.nn.relu(convolve1)

#Layer 3: Max Pooling 1: [batch_size,28,28,32]>>[batch_size,14,14,32]
#This layer reduces the number of inputs we have in the image. More concisely MP is a form of non-linear down-sampling.
#MP partitions the input image into a set of rectangles and, and then finds the maximum value for that region.
#Lets use tf.nn.max_pool function to perform max pooling.
# -Kernel Size: 2x2 (if the window is a 2x2 matrix, it would result in one output pixel). Since the first and last values are related to the batch and channels, we set this variable to 1.
# -Strides: Dictates the sliding behaviour of the kernel. In this case it will move 2 pixels everytime, thus not overlapping. The input is a matrix of size 28x28x32, and the output would be a matrix of size 14x14x32. Once again, since the first and last values are related to the batch and channels, we set this variable to 1.
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Layer 4: Convolutional Layer 2: [batch_size,14,14,32]>>[batch_size,14,14,64]
#We apply convolution again in this layer. We define a 5x5 kernel. We need 64 feature maps (implying 64 neurons).
#Notice
# -The input image is [14x14x32]. 
# -The filter is [5x5x32].
# -Since we use 64 filters of size [5x5x32], and the output of the convolutional layer would be 64 covolved image, [14x14x64]. Moreoever, the convolution result of applying a filter of size [5x5x32] on image of size [14x14x32] is an image of size [14x14x1], that is, the convolution is functioning on volume.
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
#Since we have 64 neurons, we need 64 biases for 64 outputs
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
#Convolve
convolve2= tf.add(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2)

#Layer 5: ReLU 2: [batch_size,14,14,64]>>[batch_size,14,14,64]
h_conv2 = tf.nn.relu(convolve2)

#Layer 6: Max Pooling 1: [batch_size,14,14,64]>>[batch_size,7,7,64]
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Layer 7: Fully Connected Layer 1:[batch_size,7,7,64]>>[batch_size,1024]
#You need a fully connected layer to use the Softmax and create the probabilities in the end.
#Fully connected layers take the high-level filtered images from previous layer, that is all 64 matrices, and convert them to a flat array.
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])
#So, each matrix [7x7] will be converted to a matrix of [49x1], and then all of the 64 [49x1] matrices will be connected, which make an array of size [49*64x1]=[3136x1]. We will connect it into another layer of size [1024x1]. So, the weights between these 2 layers will be [3136x1024].
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
#Which means we will have 1024 neurons and will need 1024 biases for 1024 outputs.
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
#We do the matrix product:
fcl=tf.add(tf.matmul(layer2_matrix, W_fc1), b_fc1)

#Layer 8: ReLU 3: [batch_size,1024]>>[batch_size,1024]
h_fc1 = tf.nn.relu(fcl)

#Layer 9: Drop out: [batch_size,1024]>>[batch_size,1024]
#It is a phase where the network "forgets" some features. At each training step in a mini-batch, some units get switched off randomly so that it will not interact with the network.
#That is, weights at this stage cannot be updated, nor affect the learning of the other network nodes. This can be very useful for very large neural networks to prevent overfitting. Research more about this later.
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

#Layer 10: Fully Connected Layer (Softmax): [batch_size,1024]>>[batch_size,10]
#1024 neurons
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
# 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
#Multiply the matrices
fc=tf.add(tf.matmul(layer_drop, W_fc2), b_fc2)
#Apply softmax to get the classes
y_CNN= tf.nn.softmax(fc)

#COST FUNCTION
#=============
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Train the model
#===============
sess.run(tf.global_variables_initializer())
for i in range(1100):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, float(train_accuracy)))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#Evaluate
#========
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))