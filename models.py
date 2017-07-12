import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

class EncoderModel(object):
	"""Core training model"""
	def __init__(self, max_len, hidden_units, lr):
		# Placeholders
		self.input_node = tf.placeholder(tf.float32, [None, max_len, 5])
		self.labels = tf.placeholder(tf.float32, [None])

		# weights = tf.Variable(tf.random_normal([hidden_units, 1]))
		# biases = tf.Variable(tf.random_normal([1]))

		# cell = LSTMCell(hidden_units)
		# outputs, states = tf.nn.dynamic_rnn(cell,
		# 									self.input_node,
		# 									dtype=tf.float32,
		# 									time_major=False)
		# outputs_T = tf.transpose(outputs, [1,0,2])
		# last = tf.gather(outputs_T, int(outputs_T.get_shape()[0]) - 1)
		# raw_logits = tf.matmul(last, weights) + biases
		# self.logits = tf.squeeze(tf.nn.sigmoid(raw_logits))
		# self.loss = tf.reduce_mean(tf.square(self.labels - self.logits))
		# self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

	def step(self, input_batch, label_batch, sess, valid=False):
		# if valid:
		# 	l, pred = sess.run([self.loss, self.logits],
		# 						   feed_dict={
		# 						    self.input_node: input_batch,
		# 						    self.labels: label_batch
		# 						   })
		# 	return l, pred
		# else:
		# 	l, pred, _ = sess.run([self.loss, self.logits, self.optimizer],
		# 						   feed_dict={
		# 						    self.input_node: input_batch,
		# 						    self.labels: label_batch
		# 						   })
		# 	return l, pred
		pass

