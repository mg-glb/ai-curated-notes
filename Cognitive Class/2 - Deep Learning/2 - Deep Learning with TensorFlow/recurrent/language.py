#We need to import the necessary modules for our code. We need numpy and tensorflow, obviously.
import numpy as np
import tensorflow as tf

#Now import the reader file from /resources/ptb/reader.py
import reader

#Additionally, for the sake of making it easy to play around with the model's hyperparameters, we can declare them beforehand. Feel free to change these - you'll see a difference in performance each time you change those!
#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size = 200
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 13
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 10000
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir = "resources/data/simple-examples/data"

#Some clarifications for LSTM architecture based on the arguments:
#Network Structure:
#-In this network, the number of LSTM cells are 2. To give the model more expressive power, we can add multiple layers of LSTMs to process the data. The output of the first layer will become the input of the second and so on.
#-The recurrence steps is 20, that is, our RNN is "Unfolded", the recurrence step is 20.
#-The structure is like:
#--The number of input units is 200 -> [200x200] Weights -> 200 Hidden units (first layer) -> [200x200] Weight Matrix -> 200 Hidden units (second layer) -> [200] weight matrix -> 200 unit output

#Hidden layer:
#-Each LSTM has 200 hidden units which is equivalent to the dimensionality of the embedding words and output.
#Input layer:
#-The network has 200 input cells.
#-Suppose each word is represented by an embedding vector of dimensionality e=200. The input layer of each cell will have 200 linear units. These e=200 linear units are connected to each of the h=200 LSTM units in the hidden layer (assuming there is only one hidden layer, though our case has 2 layers).
#-The input shape is [batch_size,num_steps], that is [30x20]. It will turn into [30x20x200] after embedding, and then 20x[30x200]

#Train Data
#The story starts from data:
#-Train data is a list of words, represented by numbers -N=929589 numbers.
#-We read data as mini-batch of size b=30. Assume the size of each sentence is 20 words (num_steps = 20). Then it will take int(N/b*h)+1=1548 iterations for the learner to go through all sentences once. So, the number of iterators is 1548.
#Each batch data is read from train dataset of size 600, and shape of [30x20].

#START THE SESSION
#=================
session=tf.InteractiveSession()

#PREPARE THE MODEL
#=================
#Read the data and separate it into training data, validation data and testing data.
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

#Read one mini-batch and feed the network.
itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple=itera.__next__()
x=first_touple[0]
y=first_touple[1]

#Define 2 placeholders to feed them with mini-batches, in this case x and y.
_input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
_targets = tf.placeholder(tf.int32, [batch_size, num_steps])

#Let's define a dictionary, and use it later to feed the placeholders with our first mini-batch.
feed_dict = {_input_data:x,_targets:y}

#Our goal is to create a stacked LSTM, which is a two layer LSTM network.
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

#For each LSTM, there are two state matrices, c_state and m_state. Both represent "Cell State" and "Memory State".
#Each hidden layer, has a vector of size 30, which keeps the states. So, for 200 hidden units in each LSTM, we have a matrix of size [30x200]
_initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

#Create the embeddings for the input data. Embedding is a dictionary of [10000x200] for all 10000 unique words.
embedding = tf.get_variable("embedding",[vocab_size,hidden_size])
#The function embedding_lookup goes to each row of input_data, and for each word in the row/sentence, finds the corresponding vector in embedding.
#It creates a [30x20x200] matrix, so, the first element of inputs (the first sentence), is a matrix of 20x200, which each row of it is a vector representing a word in the sentence.
#Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding, _input_data)  #shape=(30, 20, 200) 
#To construct an RNN, the function tf.nn.dynamicrnn() creates one using stacked_lstm, which is an instance of RNNCell.
#The input should be a Tensor of shape: [batch_size, max_time], in our case it would be (30,20,200)
#This method, returns a pair (outputs, new_state) where:
#-outpus is a length T list of outputs (one for each input), or a nested tuple of such elements.
#-new_state is the final state.
outputs,new_state=tf.nn.dynamic_rnn(stacked_lstm,inputs,initial_state=_initial_state)
#Let's look at the outputs. The output of the stackedLSTM comes from 200 hidden_layer, and in each step(=20), one of them gets activated. We use the linear activation to map the 200 hidden to a [?x10 matrix].
#Reshape the output tensor from [30x20x200] to [600x200]
size = hidden_size
output = tf.reshape(outputs, [-1, size])
#Create the logistic unit that will return the probability of the output word.
#Softmax = [600x200]*[200x1000]+[1x1000]->[600x1000]
softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) #[200x1000]
softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
logits = tf.matmul(output, softmax_w) + softmax_b

session.run(tf.global_variables_initializer())
logi = session.run(logits, feed_dict)
First_word_output_probablity = logi[0]

embedding_array= session.run(embedding, feed_dict)
#It's time to compare logit with target
targ = session.run(tf.reshape(_targets, [-1]), feed_dict)
first_word_target_code= targ[0]
first_word_target_vec = session.run( tf.nn.embedding_lookup(embedding, targ[0]))

#We need an objective function. Our objective is to minimize the loss function.
#That is, to minimize the average negative log probability of the target words:
#loss=−1N∑i=1Nln⁡ptargeti
#This function is already implemented and available in TensorFlow through sequence_loss_by_example so we can just use it here. The function is weighted cross-enthropy loss for a sequence of logits.
#It's arguments:
#-Logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
#-Targets: List of 1D batch-sized int32 Tensors of the same length as logits.
#-Weights: List of 1D batch-sized float Tensors of the same length as logits.
loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(_targets, [-1])],[tf.ones([batch_size * num_steps])])
#The loss function is a 1D batch-sized float Tensor [600x1]: The log-perplexity for each sequence.
cost = tf.reduce_sum(loss) / batch_size

#Now, lets store the new state as final state
final_state = new_state

#TRAIN THE MODEL
#===============
#To do Gradient Clipping in TensorFlow we have to take the following steps.
#1-Define the optimizer
#2-Extract variables that are trainable.
#3-Calculate the gradients based on the loss function.
#4-Apply the optimizer to the variables/gradients rule.

#1-Define the optimizer
#Use the GradientDescentOptimizer function to construct a new GDO. Later, we use the constructed optimizer to compute gradients for a loss and apply gradients to variables.
#Create a variable for the learning rate.
lr = tf.Variable(0.0, trainable=False)
#Create the GDO with our learning rate.
optimizer = tf.train.GradientDescentOptimizer(lr)

#2-Trainable Variables
#Defining a variable, if you passed trainable=True, the Variable() constructor automatically adds new variables to the graph collection GraphKeys.TRAINABLE_VARIABLES.
#Now, using tf.trainable_variables() you can get all variables created with trainable=True.
#Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
tvars = tf.trainable_variables()
tvars=tvars[3:]

#3-Calculate the gradients based on the loss function.
#Gradient:
#The gradient of a function is the slope of the line, or the rate of change of a function. It's a vector (a direction to move) that points in the direction of greatest increase of the function, and calculated by derivative operation.
'''#First lets recall the gradient function using a toy example
#z=(2x^2+3xy)
var_x = tf.placeholder(tf.float32)
var_y = tf.placeholder(tf.float32)
func_test = 2.0*var_x*var_x + 3.0*var_x*var_y
session.run(tf.global_variables_initializer())
feed={var_x:1.0,var_y:2.0}
print(session.run(func_test,feed))
#The tf.gradients() function allows you to compute the symbolic gradient of one tensor with respect to one or more other tensors - including variables. The tf.gradients(func,xs) function constructs symbolic partial derivatives of sum func w.r.t.x in xs.
#For the above function, the value should be d(2x^2+3xy)/dx=4x+3y
var_grad = tf.gradients(func_test,[var_x])
print(session.run(var_grad,feed))
#And d(2x^2+3xy)/dy=3x
var_grad = tf.gradients(func_test, [var_y])
print(session.run(var_grad,feed))'''
#Now let's use our variables of interest:
grad_t_list = tf.gradients(cost,tvars)
#print(session.run(grad_t_list,feed_dict)[0][0][0])
#Now we have a list of tensors, grad_t_list. We can use it to find clipped tensors. The clip_by_global_norm function clips values of multiple tensors by the ratio of the sum of their norms.
#The function has grad_t_list as its input and returns 2 things:
#-A list of clipped tensors, so called _listclipped.
#-The global norm (global_norm) of all tensors in t_list.
#Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list,max_grad_norm)
#print(session.run(grads,feed_dict)[0][0][0])

#4-Apply the optimizer to the variables to the variables / gradients tuple.
#Create the training TensorFlow Operation through our optimizer.
train_op = optimizer.apply_gradients(zip(grads,tvars))
session.run(tf.global_variables_initializer())
session.run(train_op,feed_dict)

#We learned how the model is build step by step. Now, let's then create a Class that represents our model. This class need a few things:
#-We have to create the model in accordance with our defined hyperparameters
#-We have to create the placeholders for our input data and expected outputs (the real data).
#-We have to create the LSTM cell structure and connect them with our RNN structure.
#-We have to create the word embeddings and point them to the input data.
#-We have to create the input structure for our RNN.
#-We have to instantiate our RNN model and retrieve the variable in which we should expect our outputs to appear.
#-We need to create a logistic structure to return the probability of our words.
#-We need to create the loss and cost functions for our optimizer to work, and then create the optimizer.
#-And finally, we need to create a training operation that can be run to actually train our model.
#We will do all of these steps in another file.
#CLOSE THE SESSION
#=================
session.close()