#Import TensorFlow to your project, and start a session.
import numpy as np
import tensorflow as tf
sess = tf.Session()

#Now let's imagine we want to have a RNN with stacked LSTM? For example, a two-layer LSTM. In this case, the output of the first layer will become the input of the second.
sess = tf.Session()
#4 hidden nodes = state_dim = the output_dim
LSTM_CELL_SIZE = 4
input_dim = 6
num_layers = 2
#Let's create the stacked LSTM cell.
cells = []
for _ in range(num_layers):
  cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
  cells.append(cell)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)
#Now we can create the RNN:
# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#Let's say the input sequence length is 3, and the dimensionality of the inputs is 6.
#The input should be a Tensor of shape: [batch_size,max_time,dimension], in our case it would be (2,3,6)
#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
#We can now send our input to network:
sess.run(tf.global_variables_initializer())
result = sess.run(output, feed_dict={data: sample_input})
print(result)
sess.close