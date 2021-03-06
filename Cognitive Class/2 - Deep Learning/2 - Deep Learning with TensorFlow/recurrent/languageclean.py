import tensorflow as tf
import numpy as np
import reader

#HYPERPARAMETERS
#===============
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size = 200
#The size of our vocabulary
vocab_size = 10000
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The size for each batch of data
batch_size = 30

class PTBModel(object):
    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        size = hidden_size
        self.vocab_size = vocab_size
        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM unit. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument n_hidden(size=200) of BasicLSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # Size is the same as the size of our hidden layer, and no bias is added to the Forget Gate. 
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        # Unless you changed keep_prob, this won't actually execute -- this is a dropout wrapper for our LSTM unit
        # This is an optimization of the LSTM output, but is not needed at all
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        # By taking in the LSTM cells as parameters, the MultiRNNCell function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of multiple simple cells.
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            embedding = tf.get_variable("embedding", [vocab_size, size])  #[10000x200]
            # Define where to get the data for our embeddings from
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        # Unless you changed keep_prob, this won't actually execute -- this is a dropout addition for our inputs
        # This is an optimization of the input processing and is not needed at all
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)
        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.  
        # In step 2,  second word of each of the b sentences is input in parallel. 
        # The parallelism is only for efficiency.  
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. 
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel.
        ####################################################################################################
        # Instanciating our RNN model and retrieving the structure for returning the outputs and the state #
        ####################################################################################################
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=self._initial_state)
        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(outputs, [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) #[200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b
        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                      [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        # Store the final state
        self._final_state = state
        #Everything after this point is relevant only for training
        if not is_training:
            return
        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    # Helper functions for our LSTM RNN class
    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data
    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets
    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state
    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost
    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state
    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr
    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op