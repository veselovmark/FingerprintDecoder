import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import (LSTMCell,
									GRUCell,
									MultiRNNCell,
									LSTMStateTuple)
import tensorflow.contrib.seq2seq as seq2seq

from utils import (EOS_ID,
				   PAD_ID,
				   GO_ID)

class EncoderModel(object):
	"""Core training model"""
	def __init__(self, input_shape, max_seq_len, vocab_size, cell_size, num_layers, lr=5e-5, cell_type='gru'):

		self.vocab_size = vocab_size
		self.cell_size = cell_size

		# Placeholders
		self.input_node = tf.placeholder(tf.float32, [None, input_shape])
		self.label_node = tf.placeholder(tf.float32, [max_seq_len, None, vocab_size])
		self.label_weights = tf.placeholder(tf.float32, [None, max_seq_len])

		w1 = tf.Variable(tf.random_normal([input_shape, 128]))
		b1 = tf.Variable(tf.random_normal([128]))
		w2 = tf.Variable(tf.random_normal([128, cell_size*2]))
		b2 = tf.Variable(tf.random_normal([cell_size*2]))

		dense_1 = tf.nn.relu(tf.add(tf.matmul(self.input_node, w1), b1))
		dense_2 = tf.nn.relu(tf.add(tf.matmul(dense_1, w2), b2))
		
		if cell_type == 'gru':
			cell_class = GRUCell
		elif cell_type == 'lstm':
			cell_class = LSTMCell
		else:
			raise ValueError("Cell type '%s' not valid"%cell_type)

		self.decoder_cell = single_cell = cell_class(cell_size)
		if num_layers > 1:
			self.decoder_cell = MultiRNNCell([single_cell] * num_layers)

		initial_input, initial_state = dense_2[:,:cell_size], dense_2[:,cell_size:]
		# print(initial_input.get_shape())
		state_vector = LSTMStateTuple(
    		c=initial_state,
    		h=initial_input
		)

		# batch_size, _ = tf.unstack(tf.shape(self.input_node))
		# eos_time_slice = tf.fill([batch_size], EOS_ID)
		# pad_time_slice = tf.fill([batch_size], PAD_ID)
		# go_time_slice = tf.fill([batch_size], GO_ID)

		# state_vector = initial_state
		with tf.variable_scope("run_rnn") as varscope:
			logits_list = []
			for i in range(max_seq_len):
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				time_input = self.label_node[i, :, :]
				output, state_vector = self.decoder_cell(time_input, state_vector)
				output_logits = self.project_to_chars(output)
				logits_list.append(output_logits)

		logits_tensor = tf.stack(logits_list)
		# print(logits_tensor.get_shape())
		logits_tensor_T = tf.transpose(logits_tensor, [1,0,2])
		self.softmax_logits = tf.nn.softmax(logits_tensor_T)
		label_node_T = tf.transpose(self.label_node, [1,0,2])
		# print(logits_tensor_T.get_shape())
		# print(label_node_T.get_shape())
		# print(len(self.label_weights.get_shape()))
		dense_labels = tf.argmax(label_node_T, axis=2)
		# print(dense_labels.get_shape())

		self.loss = seq2seq.sequence_loss(logits_tensor_T,
										  dense_labels,
										  self.label_weights)

		
		self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)



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

	def project_to_chars(self, input):
		weights = tf.get_variable('project_chars_weights',
								  shape=[self.cell_size, self.vocab_size],
								  initializer=tf.random_normal_initializer())
		bias = tf.get_variable('project_chars_bias',
								shape=[self.vocab_size],
								initializer=tf.random_normal_initializer())
		return tf.add(tf.matmul(input, weights), bias)

	def project_to_cell(self, input):
		weights = tf.get_variable('project_cell_weights',
								  shape=[self.vocab_size, self.cell_size],
								  initializer=tf.random_normal_initializer)
		bias = tf.get_variable('project_cell_bias',
								shape=[self.cell_size],
								initializer=tf.random_normal_initializer)
		return tf.nn.relu(tf.add(tf.matmul(input, weights), bias))

	def step(self, input_batch, label_batch, length_batch, sess, valid=False):
		if valid:
			l, pred = sess.run([self.loss, self.softmax_logits],
								   feed_dict={
								    self.input_node: input_batch,
								    self.label_node: label_batch,
		                          	self.label_weights: length_batch
								   })
		else:
			l, pred, _ = sess.run([self.loss, self.softmax_logits, self.optimizer],
								   feed_dict={
								    self.input_node: input_batch,
								    self.label_node: label_batch,
		                          	self.label_weights: length_batch
								   })
		return l, pred


	def accuracy(self, predicted_batch, true_batch):
		# correct_prediction = tf.equal(tf.argmax(), tf.argmax(valid_labels))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# raw_accuracy = sess.run(accuracy, feed_dict={x: v_prediction, y_: mnist.test.labels})
		correct_prediction = np.equal(np.argmax(predicted_batch, np.argmax(true_batch)))
		accuracy = np.mean(correct_prediction.astype(np.float32))
		return accuracy

