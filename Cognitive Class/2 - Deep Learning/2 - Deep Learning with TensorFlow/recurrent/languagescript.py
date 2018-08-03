import tensorflow as tf
import numpy as np
import reader
from languageclean import PTBModel
from languagerunepoch import run_epoch

#HYPERPARAMETERS
#===============
#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 13
#The decay for the learning rate
decay = 0.5
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir = "resources/data/simple-examples/data"

#READER OPERATIONS
#=================
#Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

#Initializes the Execution Graph and the Session
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-init_scale,init_scale)
    # Instantiates the model for training
    # tf.variable_scope add a prefix to the variables created with tf.get_variable
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True)
    #Reuses the trained parameters for the validation and testing models.
    #They are different instances but use the same variables for weights and biases.
    #They just don't change when data is input
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False)
        mtest = PTBModel(is_training=False)
    #Initialize all variables
    tf.global_variables_initializer().run()
    for i in range(max_max_epoch):
        # Define the decay for this epoch
        lr_decay = decay ** max(i - max_epoch, 0.0)
        # Set the decayed learning rate as the learning rate for this epoch
        m.assign_lr(session, learning_rate * lr_decay)
        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        # Run the loop for this epoch in the training model
        train_perplexity = run_epoch(session, m, train_data, m.train_op, verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        # Run the loop for this epoch in the validation model
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    # Run the loop in the testing model to see how effective was our training
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)