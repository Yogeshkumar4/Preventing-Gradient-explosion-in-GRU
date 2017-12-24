import tensorflow as tf
from scipy.sparse.linalg import svds
from PGE_GRU_Cell import PGE_GRUCell
import reader
import numpy as np
from scipy.sparse.linalg import svds
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
	max_grad_norm = 5
	keep_prob = 0.5
	lr_decay = 1.1
	batch_size = 20
	vocab_size = 10000
	threshold = 1.8

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

		if is_training and config.keep_prob < 1:
			output = tf.nn.dropout(output, config.keep_prob)

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

		tvars = tf.trainable_variables()
		grads = tf.gradients(self._cost, tvars)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.train.get_or_create_global_step())

		with tf.variable_scope('RNN/multi_rnn_cell/cell_0/PGE_GRUCell') as scope:
			scope.reuse_variables()
			self._w_hh = tf.get_variable('W_hh', [config.hidden_size, config.hidden_size])
			w_hh_last = tf.get_variable('W_hh_last', [config.hidden_size, config.hidden_size])
			grad = tf.subtract(w_hh_last, self._w_hh)
			w_grads = tf.scalar_mul(1/self._lr, grad)
			self._s = tf.get_variable('s', [config.hidden_size])
			# s = tf.Print(s, [s])
			self._f_norm = self._lr*tf.norm(w_grads, ord='fro', axis=(0,1))
			v_norm = tf.scalar_mul(self._f_norm - config.threshold, tf.ones([config.hidden_size]))
			s_est = tf.add(v_norm, self._s)
			self._index = tf.count_nonzero(tf.sign(s_est) + 1)
			# f_norm = tf.norm(w_grads, ord='fro', axis=(0,1))
			# w_hh = self.modify_w_hh(w_hh, w_grads, s, self._lr, config)
			# self.change_w_hh = w_hh
			# self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._cost)	
			self._new_lr = tf.placeholder(
				tf.float32, shape=[], name="new_learning_rate")
			self._lr_update = tf.assign(self._lr, self._new_lr)

			# with tf.variable_scope('PGE_GRUCell'):
			# 	w_hh = tf.get_variable('W_hh', [config.hidden_size, config.hidden_size])
			# 	s = tf.get_variable('s', [config.hidden_size])
			self._w_hh_last_update = tf.assign(w_hh_last, self._w_hh)
			self._new_w_hh = tf.placeholder(
				tf.float32, shape=[config.hidden_size, config.hidden_size], name="updating_w_hh")
			self._w_hh_update = tf.assign(self._w_hh, self._new_w_hh)

			self._new_s = tf.placeholder(
				tf.float32, shape=[config.hidden_size], name="new_s")
			self._s_update = tf.assign(self._s, self._new_s)

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

	def modify_w_hh(self, w, grad, s, lr, config):
		f_norm = tf.norm(grad, ord='fro', axis=(0,1))
		v_norm = tf.scalar_mul(lr*f_norm - config.threshold, tf.ones([config.hidden_size]))
		s_est = tf.add(v_norm, s)
		index = tf.count_nonzero(tf.sign(s_est) + 1)
		u, st, v = svds(w, k=index)
		st_count = tf.convert_to_tensor(st, dtype=tf.float32) - config.threshold
		index = tf.count_nonzero(tf.sign(st_count) + 1)
		st_len = len(st)
		with tf.variable_scope('PGE_GRUCell'):
			sc = tf.get_variable('s', [config.hidden_size])
			sc = tf.concat(tf.convert_to_tensor(reversed(st), dtype=tf.float32), tf.slice(sc,[st_len],[config.hidden_size - st_len]))
		submat = np.zeros((config.hidden_size, config.hidden_size))
		u = map(list, zip(*u))
		for i in range(index):
			submat += (st[i]-2)*np.dot(np.expand_dims(u[i], axis=1),np.expand_dims(v[i], axis=0))
		s_matrix = tf.convert_to_tensor(submat, dtype=tf.float32)
		return tf.subtract(w, s_matrix)	


	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	def assign_w_hh(self, session, w_hh_value):
		session.run(self._w_hh_last_update)
		session.run(self._w_hh_update, feed_dict={self._new_w_hh: w_hh_value})
		
	def assign_s(self, session, s_value):
		session.run(self._s_update, feed_dict={self._new_s: s_value})		

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

	@property
	def w_hh(self):
		return self._w_hh	

	@property
	def index(self):
		return self._index	

	@property
	def s(self):
		return self._s	

	@property
	def f_norm(self):
		return self._f_norm			
	

def run_epoch(session, model, config, eval_op=None, verbose=False):
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
		fetches["eval_op"] = eval_op[0]
		fetches["w_hh"] = eval_op[1]
		fetches["index"] = eval_op[2]
		fetches["s"] = model.s
		fetches["f_norm"] = model.f_norm

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (h) in enumerate(model.initial_state):
			feed_dict[h] = state[i]

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		try:
			w = vals["w_hh"]
			index = vals["index"]
		except KeyError:
			index = 0	
		# print(vals["s"][:10])
		# print(cost)
		if index > 0:
			u, st, v = svds(w, k=min(index, config.hidden_size - 1))
			st_count = st - config.threshold
			index = np.count_nonzero(np.sign(st_count) + 1)
			st_len = len(st)
			vals["s"] += [vals["f_norm"]]*len(vals["s"])
			sc = list(reversed(st)) + list(vals["s"])[st_len:]
			submat = np.zeros((config.hidden_size, config.hidden_size))
			u = list(map(list, zip(*u)))
			for i in range(index):
				submat += (st[i]-2)*np.dot(np.expand_dims(u[i], axis=1),np.expand_dims(v[i], axis=0))
			modified_w = w - submat	
			sc.sort(reverse=True)
			model.assign_w_hh(session, modified_w)
			model.assign_s(session, sc)

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

		# for op in tf.all_variables():
		# 	print str(op.name) 
	
		sv = tf.train.Supervisor(logdir="./log/")
		config_proto = tf.ConfigProto(allow_soft_placement=False)
		with sv.managed_session(config=config_proto) as session:
			lr_decay = 1
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay*lr_decay 
				if i < config.max_epoch:
					lr_decay = 1.0
				m.assign_lr(session, config.learning_rate/lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, config, eval_op=[m.train_op, m.w_hh, m.index],
											 verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid, config)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			test_perplexity = run_epoch(session, mtest, config)
			print("Test Perplexity: %.3f" % test_perplexity)	

if __name__ == "__main__":
	tf.app.run()				