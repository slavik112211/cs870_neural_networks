
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
OUTPUT_LAYER_SIZE = trains_dataset.OUTPUT_LAYER_SIZE

# from tensorflow.examples.tutorials.mnist import input_data
# INPUT_LAYER_SIZE = 784
# OUTPUT_LAYER_SIZE = 10

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = None

def main(_):
  # pdb.set_trace()
  # Import data
  dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, reshape=False)

  # Create the model
  x = tf.placeholder(tf.float32, [None, INPUT_LAYER_SIZE])
  W = tf.Variable(tf.zeros([INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE]))
  b = tf.Variable(tf.zeros([OUTPUT_LAYER_SIZE]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_SIZE])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # tf.summary.image('input', )
  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.histogram('weights', W)
  tf.summary.histogram('biases', b)
  tf.summary.histogram('activations', y)
  merged_summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter("/tmp/tensorflow_summary/")

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  summary_writer.add_graph(sess.graph)

  epochs_completed = 0; epochs_completed_actual = 0
  # Train
  for i in range(72000): # 40000*10 / 1197 = 334 epochs
    batch_xs, batch_ys, epochs_completed_actual = dataset.train.next_batch(10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if epochs_completed != epochs_completed_actual:
      epochs_completed = epochs_completed_actual
      if epochs_completed != 0 and (epochs_completed % 5 == 0 or epochs_completed == 60 or i == 71999):
        s = sess.run(merged_summary, feed_dict={x: dataset.test.images, y_: dataset.test.labels})
        summary_writer.add_summary(s, epochs_completed)
        print(sess.run(cross_entropy, feed_dict={x: dataset.test.images, y_: dataset.test.labels}))
        print(sess.run(accuracy, feed_dict={x: dataset.test.images, y_: dataset.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./mnist',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
