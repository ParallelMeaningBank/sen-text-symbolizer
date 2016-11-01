# -*- coding: utf-8 -*-
'''
@project: SEN text symbolizer - Parallel meaning bank - RUG

@author: duynd

@acknowledgement: inspired by Aymeric Damien tutorial on Tensorflow
https://github.com/aymericdamien/TensorFlow-Examples

The program let user input a string spelling out a number, then return the cor-
rresponding number.

'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import RNNFlags,SpelledNumber
from functions import *

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 26 # text feature vector input
n_steps = 2 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1000 # MNIST total classes (0-9 digits)

FLAGS = RNNFlags.FLAGS(checkpoint_dir="./tmp/");

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#def str2num():
#    vct = numberStringExtract(inp).reshape((1, n_steps, n_input))
#    print(inp)
#    print(type(vct)," ", vct.shape," ",vct.dtype)
#    predictions = sess.run(pred, feed_dict={x: vct}) 
#    print("- Result for string " + inp + " is ", np.argmax(predictions))

if __name__=='__main__':
    
    pred = RNN(x, weights, biases)
       
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Simple sample predict
    #sample_predict = tf.argmax(pred,1);
    sample_predict = tf.argmax(pred,1);
    
    # Initializing the variables
    init = tf.initialize_all_variables()
    
    # defaults to saving all variables - in this case weights and biases
    saver = tf.train.Saver()  
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        print("Now we test an customized sample. Enter 'no' to quit");
        print("-> Loading model...");
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("/!\ [ERR] Model not found")            
        
        inp = "";        
        while 1:
            inp = raw_input("Enter the text: ")
            if (inp=="no" or inp==""): 
                print("Good bye!")
                break;
            res = sequenceFetch(inp); # Fetch string into SpelledNumber object
            
            # Transform the text to number
            ## Identify the missing level
            txts = res.texts;            
            flgs = res.flags;
            
            if not (flgs.trillion):
                 # Extract feature
                vct = numberStringExtract(txts.trillion).reshape((1, n_steps, n_input))
                txts.trillion = str(np.argmax(sess.run(pred, feed_dict={x: vct})))
            if not (flgs.billion):
                 # Extract feature
                vct = numberStringExtract(txts.billion).reshape((1, n_steps, n_input))
                txts.billion = str(np.argmax(sess.run(pred, feed_dict={x: vct})))
            if not (flgs.million):
                 # Extract feature
                vct = numberStringExtract(txts.million).reshape((1, n_steps, n_input))
                txts.million = str(np.argmax(sess.run(pred, feed_dict={x: vct})))
            if not (flgs.thousand):
                vct = numberStringExtract(txts.thousand).reshape((1, n_steps, n_input)) # Extract feature
                txts.thousand = str(np.argmax(sess.run(pred, feed_dict={x: vct})))
            if not (flgs.unit):
                vct = numberStringExtract(txts.unit).reshape((1, n_steps, n_input)) # Extract feature
                txts.unit = str(np.argmax(sess.run(pred, feed_dict={x: vct})))
            # Now update the SpelledNumber
                
            res = SpelledNumber.SpelledNumber(txts.trillion,txts.billion,txts.million,txts.thousand,txts.unit);
            print(res.toString());
            print("-> Result value: ",int(res.actualValue),"\n");
            