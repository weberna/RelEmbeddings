#########################################################
#       Define and run the neural network for relation embeddings
#########################################################


# pylint: disable=missing-docstring
import argparse
import os.path
import pickle
import sys
import numpy as np  
import time
from tuple_reader import TupleData
import hotvector
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Basic model parameters as external flags.
FLAGS = None

#EMBED_SIZE = 100
EMBED_SIZE = 150

INIT_FILE = "initial_weights.pkl"
#INIT_FILE = "initial_weights_normalized.pkl"


REL_COUNT = 46747
WORD_COUNT = 113828


def placeholder_inputs(hot3_dim, batch_size):
    context1_ph = tf.placeholder(tf.float32, shape=(batch_size,hot3_dim))
    context2_ph = tf.placeholder(tf.float32, shape=(batch_size,hot3_dim))
    #target_placeholder = tf.placeholder(tf.int32, shape=(batch_size,hot3_dim))
    target_arg1_ph = tf.placeholder(tf.int32, shape=(batch_size,WORD_COUNT))
    target_rel_ph = tf.placeholder(tf.int32, shape=(batch_size,REL_COUNT))
    target_arg2_ph = tf.placeholder(tf.int32, shape=(batch_size,WORD_COUNT))

#    return context1_ph, context2_ph, target_placeholder
    return context1_ph, context2_ph, target_arg1_ph, target_rel_ph, target_arg2_ph


def infer_target(context1_ph, context2_ph, hot3_dim):
    """
    Given place holders for the first two context windows and the hot3_dim,
    return back the computation that embedds the context and return the result of 
    decoding the average of the embedding
    Return back the embedding matrix as well
    """
    #encoding matrix, this matrix contains the embeddings
 #   encode_mat = tf.Variable(
 #       tf.truncated_normal([hot3_dim, EMBED_SIZE],
 #                           stddev=1.0 / math.sqrt(float(hot3_dim))))

    weights = pickle.load(open(INIT_FILE, 'rb'))
    weights = weights.astype(np.float32)
    encode_mat = tf.Variable(weights, expected_shape=[hot3_dim, EMBED_SIZE])

                          
   
#    decode_mat = tf.Variable(tf.truncated_normal([EMBED_SIZE, hot3_dim],stddev=1.0 / math.sqrt(float(EMBED_SIZE))))
    decode_arg1_mat = tf.Variable(tf.truncated_normal([EMBED_SIZE, WORD_COUNT],stddev=1.0 / math.sqrt(float(EMBED_SIZE))))
    decode_rel_mat = tf.Variable(tf.truncated_normal([EMBED_SIZE, REL_COUNT],stddev=1.0 / math.sqrt(float(EMBED_SIZE))))
    decode_arg2_mat = tf.Variable(tf.truncated_normal([EMBED_SIZE, WORD_COUNT],stddev=1.0 / math.sqrt(float(EMBED_SIZE))))

    #biases = tf.Variable(tf.zeros([hot3_dim]))
    biases_a1 = tf.Variable(tf.zeros([WORD_COUNT]))
    biases_rel = tf.Variable(tf.zeros([REL_COUNT]))
    biases_a2 = tf.Variable(tf.zeros([WORD_COUNT]))

    embed1 = tf.matmul(context1_ph, encode_mat)
    embed2 = tf.matmul(context2_ph, encode_mat)
    average_embed = (embed1 + embed2) / 2


    
    #output = tf.matmul(average_embed, decode_mat) + biases
    a1_output = tf.matmul(average_embed, decode_arg1_mat) + biases_a1
    rel_output = tf.matmul(average_embed, decode_rel_mat) + biases_rel
    a2_output = tf.matmul(average_embed, decode_arg2_mat) + biases_a2
#    return (output, encode_mat)
    return a1_output, rel_output, a2_output, encode_mat



def loss(a1_output, rel_output, a2_output, a1_target_ph, rel_target_ph, a2_target_ph):
    """Calculates the loss from the output and the target.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    
    a1_target_ph = tf.to_int64(a1_target_ph)
    rel_target_ph = tf.to_int64(rel_target_ph)
    a2_target_ph = tf.to_int64(a2_target_ph)

    a1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a1_output, a1_target_ph))
    rel_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(rel_output, rel_target_ph))
    a2_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a2_output, a2_target_ph))

    loss_avg = (a1_cross_entropy + rel_cross_entropy + a2_cross_entropy) / 3

    #target_ph = tf.to_int64(target_ph)
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output, target_ph)
    #loss = tf.reduce_mean(cross_entropy)
    
    

    return loss_avg


def training(loss, learning_rate):
    """
    Return the training operation given the loss operation and the 
    learning weight
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss)
    return train_op


def fill_feed_dict(tuple_data, c1ph, c2ph, a1_target_ph, rel_target_ph, a2_target_ph, sample_index, batch_size):
    """
    Pass in TupleData object tuple_data,
    and context 1, context2, context3 placeholders
    as well as the sample index
    """

    context1, context2, target = tuple_data.get_sample(sample_index, batch_size)
    a1 = target[:, :WORD_COUNT]
    rel = target[:,WORD_COUNT:WORD_COUNT + REL_COUNT]
    a2 = target[:, WORD_COUNT + REL_COUNT:]
    #context1 = numpy.array([context1])
    #context2 = numpy.array([context2])
    #target = numpy.array([target])
    feed_dict = {
      c1ph: context1,
      c2ph: context2,
      a1_target_ph: a1,
      rel_target_ph: rel,
      a2_target_ph: a2
    }
    return feed_dict



def run_training():

    
    #load tuple data
    tuple_data = pickle.load(open("TupleDataOut.pkl", "rb"))
    
    hot3_dim = tuple_data.tuples[0].shape[1]
    #hot3_dim = tuple_data.tuples.shape[1]

#    learning_rate = 1.0
    learning_rate = 0.5

    batch_size = 100

    with tf.Graph().as_default():
        con1ph, con2ph, a1_target_ph, rel_target_ph, a2_target_ph  = placeholder_inputs(hot3_dim, batch_size)

        #computation to guess the target embedding given the context
        a1_output, rel_output, a2_output, embeddings = infer_target(con1ph, con2ph, hot3_dim)
#        outputs = tf.Print(outputs, [outputs], "Outputs: ") 

        #Get the loss for the above
        loss_op = loss(a1_output, rel_output, a2_output,a1_target_ph, rel_target_ph, a2_target_ph)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_op, learning_rate)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        sess.run(init)

        # Start the training loop.
        #for step in xrange(FLAGS.max_steps):
        EPOCHS = 2
        iteration = 0
        avg_loss = 0
        batches = tuple_data.windows // batch_size
        for i in range(EPOCHS):
            order = np.arange(tuple_data.windows // batch_size)
            np.random.shuffle(order)
            print(len(order))
            #for step in order[:9000]:
            for step in order:

#                print(embeddings.eval(session=sess))
#                print("------------------------------")

                
      #          print(embeddings.eval(session=sess))
      #          print("--------------------------------")

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                feed_dict = fill_feed_dict(tuple_data, con1ph, con2ph, a1_target_ph, rel_target_ph, a2_target_ph, step, batch_size)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
    #            if step % 2000 == 0:
    #                print(con1ph.eval(session=sess))


                _, loss_value = sess.run([train_op, loss_op],
                                       feed_dict=feed_dict)

                print(loss_value)
                avg_loss = avg_loss + loss_value
                if iteration % 100 == 0:
                    avg_loss = avg_loss / 100
                    print("Iteration: {}, Loss {}".format(iteration, avg_loss))
                    print(embeddings.eval(session=sess))
                    print("------------------------------")
                    avg_loss = 0
                iteration = iteration + 1

        print(embeddings.eval(session=sess))
        pickle.dump(embeddings.eval(session=sess), open('OUTPUT_EMBED.pkl', 'wb'))



if __name__ == "__main__":
    run_training()
