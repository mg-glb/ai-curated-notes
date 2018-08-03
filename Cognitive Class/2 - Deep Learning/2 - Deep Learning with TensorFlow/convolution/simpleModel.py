#Import TensorFlow and the MNIST dataset.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#From the dataset, get the data with One-Hot Encoding activated.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Interactive sessions allow you to create graphs on the fly.
sess = tf.InteractiveSession()

#The x placeholder represents the space allocated input or images. Both input and ouput tensors are 2-D.
# Each input has 784 pixels distributed by a 28 width x 28 height matrix.
# The 'shape' argument defines the tensor size by its dimensions.
# -1st dimension = None. Indicates that the batch size, can be of any size.
# -2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image (28x28).
x  = tf.placeholder(tf.float32, shape=[None, 784])
#The y placeholder represents the final output, the classes, labels. In our case the zero to nine range.
# The 'shape' argument defines the tensor size by its dimensions.  
# 1st dimension = None. Indicates that the batch size, can be of any size.
# 2nd dimension = 10. Indicates the number of targets/outcomes.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight tensor
# The shape argument in the weight tensor is what transforms the input into the output.
# -1st dimension = 784. The size of each image.
# -2nd dimension = 10. The number of labels we want to classify to.
W = tf.Variable(tf.zeros([784,10],tf.float32))
# Bias tensor
# The bias shifts the result of each variable to either the left or the right.
# In this case we don't want that, so we set the bias to zero.
b = tf.Variable(tf.zeros([10],tf.float32))
# Initialize the variables.
sess.run(tf.global_variables_initializer())

#Creating the neurons
#====================
#We will create 10 neurons with 784 weights each.
#Each column in W is a neuron. Each row in W is a list of weights to apply to X.
#When x and W are multiplied, the result is a raw probability for each of the ten classes. But this is still not enough to determine whether we got the right digit or not.
raw=tf.matmul(x,W)
#This is when the biases come into play. In a more fundamental sense, the bias is information that know about each class that must be added for the model to produce an accurate result.
#Each class has one bias. That is why b has size 10. The first bias is added to the first neuron, and so on.
biased=tf.add(raw,b)
#Once we get the array with the probabilities for each class, we use softmax to get the most probable output. That is our prediction. A key note about soft
y=tf.nn.softmax(biased)

#Create the cost function
#========================
#We first have to define the cross_entropy between the real distribution "y_" and the unnatural distribution "y".
#1- Multiply the output placeholder by the logarithm of the computed output.
# The result will be a matrix of height equal to the number of inputs, and length equal to ten.
yLogy = y_ * tf.log(y)
#2- We first get the sum of the values of the vector we got in the previous function.
# The result is a column vector of height equal to the number of inputs.
ySum = -tf.reduce_sum(yLogy, axis=1)
#3- Get the mean of the ySum. The result is a scalar.
cross_entropy = tf.reduce_mean(ySum)
#4-The reason we needed the cross_entropy is that we will be using it as the reduction method for the model we will use:
# The Gradient Descent Optimizer. Some time in the future I'll go into the Math of the GDO, but for now, look it as
# the model that tries to look for either local minima or maxima using derivatives and a parameter known as the Learning
# Rate. NOTE: See how the GDO and cross_entropy are related.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Model Training
#==============
#Now it's time to train the model, using the cross_entropy variable we have. Note that the real value of this variable is:
# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(tf.add(tf.matmul(x,W),b))), axis=1))
#If you look closely, W and b are null tensors, while y_ and x are place holders.
#That means that we have to feed the model by providing input to these variables.
#So, from the training set of MNIST, we iterate a thousand times. Each time we will feed the model with 50 images.
#The mnist.train.next_batch(i) function will return a Python tuple.
# -The first element contains a ix784 matrix, which are the normalized images.
# -The second element contains a ix10 matrix, which are the classes for each digit.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#And that's it! Our model is trained! Notice that TensorFlow saves us from the ugly task of programming the back-prop algorithm. If you were using Theano or other library, you would have around some hundred lines more. Now let's test the accuracy of our new model.
#Remember that argmax returns the index that contains the largest element of an array. The axis=1 is done to produce
# a column vector that will contain the prediction accuracy of each input. This is passed to correct_prediction.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#In order to get the total accuracy, we cast the results to float, and then reduce to a scalar.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Note however that "accuracy" is a placeholder! If we try to evaluate it with the same value we used before we will get
# 100% percent accuracy (which may not be true).
#To actually evaluate, we need to give the already trained model some test data, to actually evaluate the model.
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

#At the end, close the session
sess.close()