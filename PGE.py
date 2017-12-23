import tensorflow as tf
from scipy.sparse.linalg import svds
from PGE_GRU_Cell import PGE_GRUCell
import reader
import numpy as np
import time

class PTBInput(object):
	"""The input data."""

	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(
			data, batch_size, num_steps, name=name)

	
class Config(object):
	"""config."""
	learning_rate = 1.0
	num_layers = 1
	num_steps = 35
	hidden_size = 650
	max_epoch = 10
	max_max_epoch = 75
	keep_prob = 0.5
	lr_decay = 1.1
	batch_size = 20
	vocab_size = 10000


class PGEModel(object):
	def __init__(self, is_training, config, input_):
		self._is_training = is_training
		self._input = input_
		self._rnn_params = None
		self._cell = None
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
		inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		inputs = inputs*0.01

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		output, state = self._build_rnn_graph_gru(inputs, config, is_training)

		softmax_w = tf.get_variable(
			"softmax_w", [size, vocab_size], dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
		logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
		 # Reshape logits to be a 3-D tensor for sequence loss
		logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

		# Use the contrib sequence loss and average over the batches
		loss = tf.contrib.seq2seq.sequence_loss(
			logits,
			input_.targets,
			tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True)

		# Update the cost
		self._cost = tf.reduce_sum(loss)
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)

		self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._cost)	
		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def _build_rnn_graph_gru(self, inputs, config, is_training):
		"""Build the inference graph using canonical LSTM cells."""
		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		def make_cell():
			cell = PGE_GRUCell(config.hidden_size, config.hidden_size)	
			# if is_training and config.keep_prob < 1:
			# 	cell = tf.contrib.rnn.DropoutWrapper(
			# 		cell, output_keep_prob=config.keep_prob)
			return cell

		cell = tf.contrib.rnn.MultiRNNCell(
			[make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

		self._initial_state = tuple([tf.zeros([config.batch_size, config.hidden_size], tf.float32)]*config.num_layers)
		state = self._initial_state
		# Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
		# This builds an unrolled LSTM for tutorial purposes only.
		# In general, use the rnn() or state_saving_rnn() from rnn.py.
		#
		# The alternative version of the code below is:
		#
		# inputs = tf.unstack(inputs, num=num_steps, axis=1)
		# outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
		#                            initial_state=self._initial_state)
		outputs = []
		with tf.variable_scope("RNN"):
		  for time_step in range(self.num_steps):
			if time_step > 0: tf.get_variable_scope().reuse_variables()
			t_inp = inputs[:, time_step, :]
			(cell_output, state) = cell(t_inp, state)
			outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
		return output, state	

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op
	

def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (h) in enumerate(model.initial_state):
			feed_dict[h] = state[i]

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
				 iters * model.input.batch_size /
				 (time.time() - start_time)))

	return np.exp(costs / iters)	

def main(_):
	raw_data = reader.ptb_raw_data("./data/")
	train_data, valid_data, test_data, _ = raw_data

	config = Config()
	eval_config = Config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default():

		with tf.name_scope("Train"):
			train_input = PTBInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None):
				m = PGEModel(is_training=True, config=config, input_=train_input)

		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True):
				mvalid = PGEModel(is_training=False, config=config, input_=valid_input)

		with tf.name_scope("Test"):
			test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True):
				mtest = PGEModel(is_training=False, config=eval_config,
						 input_=test_input)

		models = {"Train": m, "Valid": mvalid, "Test": mtest}

	
		sv = tf.train.Supervisor(logdir="./log/")
		config_proto = tf.ConfigProto(allow_soft_placement=False)
		with sv.managed_session(config=config_proto) as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay 
				if i < config.max_epoch:
					lr_decay = 1.0
				m.assign_lr(session, config.learning_rate/lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op,
											 verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)	

if __name__ == "__main__":
	tf.app.run()				