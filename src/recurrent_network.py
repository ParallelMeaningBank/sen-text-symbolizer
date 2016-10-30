# -*- coding: utf-8 -*-
'''
@project: SEN text symbolizer - Parallel meaning bank - RUG

@author: duynd

@acknowledgement: inspired by Aymeric Damien tutorial on Tensorflow
https://github.com/aymericdamien/TensorFlow-Examples
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle,sys,Datasets,functions,RNNFlags

# Parameters
learning_rate = 0.001
training_iters = 200000
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


if __name__=='__main__':
    if (sys.argv[1]!=''):
        pred = RNN(x, weights, biases)
    
        # Load the data from pickle
        txt2numData = pickle.load( open( sys.argv[1], "rb" ) )
        
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
        
        saver = tf.train.Saver()  # defaults to saving all variables - in this case weights and biases
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = txt2numData.train.next_batch(batch_size)
                # Reshape data to get 28 seq of 28 elements
                #print(batch_x.shape)
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                #print(type(batch_x)," ", batch_x.shape," ",batch_x.dtype)
                #print(batch_x.shape)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))  
                step += 1
                # Save model
                saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=step)  
            print("Optimization Finished and saved model!")
            
            # Calculate accuracy for test vectors
            test_len = txt2numData.test.texts.shape[0]
            test_data = txt2numData.test.texts[:test_len].reshape((-1, n_steps, n_input))
            test_label = txt2numData.test.labels[:test_len]
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
            # Try testing single samples
            print("Now we test an customize sample. Enter 'no' to quit");
            print("-> Loading model...");
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("/!\ [ERR] Model not found")            
            
            inp = "";        
            while 1:
                inp = raw_input("Enter the text: ")
                if (inp=="no"): 
                    print("Good bye!")
                    break;
                if inp!="":
                    vct = functions.numberStringExtract(inp).reshape((1, n_steps, n_input))
                    print(inp)
                    print(type(vct)," ", vct.shape," ",vct.dtype)
                    predictions = sess.run(pred, feed_dict={x: vct})
                    
                    print("- Result for string " + inp + " is ", np.argmax(predictions))