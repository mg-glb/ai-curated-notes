#Import TensorFlow to your project, and start a session.
import numpy as np
import tensorflow as tf
sess = tf.Session()

#We want to create a network that has only one LSTM cell.
#We have to pass 2 elements to LSTM, the prv_output and prv_state, so called, h and c.
#Therefore, we initialize a state vector, state. Here, state is a tuple with 2 elements, each one is of size [1x4], one for passing prv_output to the next time step, and another for passing the prv_state to next time stamp.
#Output size (dimension), which is same as hidden size in the cell
LSTM_CELL_SIZE = 4
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2

#Let's define a sample input. In this example, batch_size=2, and seq_len=6.
sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]],dtype=tf.float32)
#Now we can pass the input to lstm_cell, and check the new state:
with tf.variable_scope("LSTM_sample1"):
    output,state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
#Print the output
print (sess.run(output))
sess.close()