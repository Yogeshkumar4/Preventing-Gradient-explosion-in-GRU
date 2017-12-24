import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
import numpy as np

np.set_printoptions(threshold=np.nan)

class PGE_GRUCell(tf.contrib.rnn.RNNCell):
	""" GRUCell for Preventing Gradient Explosions in Gated Recurrent Units 
	(https://papers.nips.cc/paper/6647-preventing-gradient-explosions-in-gated-recurrent-units.pdf) """

	def __init__(self, num_units, inp_size):
		self.num_units = num_units
		self.inp_size = inp_size
		self.s, self.u, v = tf.svd(tf.random_normal([self.num_units, self.num_units]), full_matrices=True)

	@property
	def output_size(self):
		return self.num_units

	@property
	def state_size(self):
		return (self.num_units)
		
	def __call__(self, inputs, state):
		with tf.variable_scope('PGE_GRUCell'):
			W_xz = tf.get_variable('W_xz', initializer=tf.random_normal([self.num_units, self.inp_size], stddev = 1.0/650))
			W_hz = tf.get_variable('W_hz', initializer=tf.random_normal([self.num_units, self.num_units], stddev = 1.0/650))
			W_xr = tf.get_variable('W_xr', initializer=tf.random_normal([self.num_units, self.inp_size], stddev = 1.0/650))
			W_hr = tf.get_variable('W_hr', initializer=tf.random_normal([self.num_units, self.num_units], stddev = 1.0/650))
			W_xh = tf.get_variable('W_xh', initializer=tf.random_normal([self.num_units, self.inp_size], stddev = 1.0/650))	
			W_hh = tf.get_variable('W_hh', initializer=self.u)
			W_hh_last = tf.get_variable('W_hh_last', initializer=self.u, trainable=False)
			sing_val = tf.get_variable('s', initializer=self.s, trainable=False)

		zt = tf.sigmoid(tf.matmul(inputs, W_xz) + tf.matmul(state, W_hz))
		rt = tf.sigmoid(tf.matmul(inputs, W_xr) + tf.matmul(state, W_hr))

		h_hat = tf.tanh(tf.matmul(inputs, W_xh) + tf.matmul(rt*state, W_hh))
		h = zt*state + (1-zt)*h_hat

		return h, (h)	

					