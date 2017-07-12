import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import (LSTMCell,
									GRUCell,
									MultiRNNCell)
import tensorflow.contrib.seq2seq as seq2seq

class EncoderModel(object):
	"""Core training model"""
	def __init__(self, input_shape, max_seq_len, vocab_size, cell_size, num_layers, lr=5e-5, cell_type='gru'):
		# Placeholders
		self.input_node = tf.placeholder(tf.float32, [None, input_shape])
		self.label_node = tf.placeholder(tf.int8, [None, max_seq_len, vocab_size])

		w1 = tf.Variable(tf.random_normal([input_shape, 128]))
		b1 = tf.Variable(tf.random_normal([128]))
		w2 = tf.Variable(tf.random_normal([128, 64]))
		b2 = tf.Variable(tf.random_normal([64]))

		layer_1 = tf.nn.relu(tf.add(tf.matmul(self.input_node, w1), b1))
		layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), b2))

		if cell_type == 'gru':
			cell_class = GRUCell
		elif cell_type == 'lstm':
			cell_class = LSTMCell
		else:
			raise ValueError("Cell type '%s' not valid"%cell_type)

		self.cell = single_cell = cell_class(cell_size)
		if num_layers > 1:
			self.cell = MultiRNNCell([single_cell] * num_layers)

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

