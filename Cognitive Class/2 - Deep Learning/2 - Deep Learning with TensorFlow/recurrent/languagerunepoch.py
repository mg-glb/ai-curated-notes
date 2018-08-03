import tensorflow as tf
import numpy as np
import time
import reader

##########################################################################################################################
# run_epoch takes as parameters the current session, the model instance, the data to be fed, and the operation to be run #
##########################################################################################################################
def run_epoch(session, m, data, eval_op, verbose=False):
    #Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    #m.initial_state = tf.convert_to_tensor(m.initial_state) 
    #state = m.initial_state.eval()
    state = session.run(m.initial_state)
    #For each step and data point
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        #Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += cost
        #Add number of steps to iteration counter
        iters += m.num_steps
        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
              iters * m.batch_size / (time.time() - start_time)))
    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)