
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import trains_dataset
import trains_tensorflow_dataset as input_data
INPUT_LAYER_SIZE = trains_dataset.INPUT_LAYER_SIZE
HIDDEN_LAYER_SIZE = 30
OUTPUT_LAYER_SIZE = trains_dataset.OUTPUT_LAYER_SIZE

# from tensorflow.examples.tutorials.mnist import input_data
# INPUT_LAYER_SIZE = 784
# OUTPUT_LAYER_SIZE = 10

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

def main():
  # pdb.set_trace()
  # with tf.device('/gpu:0'):
    dataset = input_data.read_data_sets(None, one_hot=True, reshape=False)

    x = tf.placeholder(tf.float32, [None, INPUT_LAYER_SIZE])
    labels = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_SIZE])

    hidden_weights = tf.Variable(tf.random_normal([INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE]))
    hidden_biases = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE]))
    out_weights = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE]))
    out_biases = tf.Variable(tf.random_normal([OUTPUT_LAYER_SIZE]))

    hidden_layer_activations = tf.nn.relu(tf.add(tf.matmul(x, hidden_weights), hidden_biases))
    out_layer = tf.matmul(hidden_layer_activations, out_weights) + out_biases

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out_layer))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # tf.summary.image('input', )
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    # tf.summary.histogram('weights', W)
    # tf.summary.histogram('biases', b)
    # tf.summary.histogram('activations', y)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("/tmp/tensorflow_summary_1_hidden/")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    summary_writer.add_graph(sess.graph)

    epochs_completed = 0; epochs_completed_actual = 0
    # Train
    for i in range(40000): # 40000*10 / 1197 = 334 epochs
      batch_xs, batch_ys, epochs_completed_actual = dataset.train.next_batch(10)
      sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})
      # if epochs_completed != epochs_completed_actual:
      #   epochs_completed = epochs_completed_actual
      #   print(sess.run(accuracy, feed_dict={x: dataset.test.images, labels: dataset.test.labels}))
      if i % 4000 == 0 or i == 79999:
        s = sess.run(merged_summary, feed_dict={x: batch_xs, labels: batch_ys})
        summary_writer.add_summary(s, i)
        print(sess.run(accuracy, feed_dict={x: dataset.test.images, labels: dataset.test.labels}))

    print(sess.run(accuracy, feed_dict={x: dataset.test.images, labels: dataset.test.labels}))

main()