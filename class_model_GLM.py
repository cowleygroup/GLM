# Class for generalized linear model
#	where y = g(Beta^T * X) + epsilon_noise
#		Beta is a vector of linear weights
#		g(.) is a link function (pointwise nonlinearity)
#
#	For a given dataset X and y, you can train a GLM and test it.
#	Note: This code does not consider output probability distributions (e.g., Poisson),
#		which a full GLM could include.
#
#   GLM is trained with stochastic gradient descent (SGD) in tensorflow;
#	this allows for training on large datasets.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model


class GLMModelClass: 


	def __init__(self, num_input_vars, link_function='linear', learning_rate=1e-3, mean_output_data=0, stddev_output_data=1):
		# Class initialization
		#
		# INPUT:
		#	num_input_vars: (int), number of input/feature variables for X
		#	link_function: ('linear', 'relu', 'sigmoid'), defines the link/pointwise nonlinear/activation function between Beta^T * X and y
		#	learning_rate: (float), learning rate for SGD
		#	mean_output_data: (float), initial value for the output offset parameter (can be set to be equal the mean(y_train)); helps for optimization
		#	stddev_output_data: (float), initial value for the output standard deviation (can be equal to stddev(y_train)); helps for optimization
		#
		# OUTPUT:
		#	None.

		# define tensorflow model for GLM
		if True:
			x_input = Input(shape=(num_input_vars,), name='input')
			x = Dense(units=1, name='linear_weights')(x_input)  # linear weights
			x = BatchNormalization(axis=-1, name='batchnorm')(x)  # normalizes input
			x = Activation(activation=link_function, name='act')(x)  # activation function
			x = Dense(units=1, name='recenter')(x)  # match target data's mean and std deviation

			self.model = Model(inputs=x_input, outputs=x)

			optimizer = SGD(learning_rate=learning_rate, momentum=0.7)
			self.model.compile(optimizer=optimizer, loss='mean_squared_error')

		# initialize weights for easier optimization
		if True:
			weights = self.model.get_weights()

			weights[0][:,0] = 1/num_input_vars # linear weights set to average
			weights[1][0] = 0.
			weights[6][0,0] = mean_output_data
			weights[7][0] = stddev_output_data
			self.model.set_weights(weights)
		
		self.num_input_vars = num_input_vars


	def train_model(self, X_train, y_train, num_passes=100):
		# Trains GLM on training data
		#
		# INPUT:
		#   X_train: (num_input_vars, num_training_samples), input data to train on (i.e., features)
		#   y_train: (num_training_samples,), output data to train on (i.e., targets)
		#	num_passes: (int), number of gradient steps to take on this batch of data
		#
		# OUTPUT:
		#	None.  GLM's weights/predictions can be accessed after training.

		# error checking
		if X_train.shape[0] != self.num_input_vars:
			raise ValueError('num input vars in X_train not equal to defined num input vars.')
		elif y_train.ndims > 1:
			raise ValueError('y_train should be a one-dimensional vector.')
		elif X_train.shape[1] != y_train.size:
			raise ValueError('X_train and y_train need to have the same number of samples.')

		for ipass in range(num_passes):
			self.model.train_on_batch(X_train.T, y_train)


	def get_weights(self):
		# Gets the weights/parameters of the GLM
		#
		# INPUT:
		#	None.
		#
		# OUTPUT:
		#	linear_weights: (num_input_vars,), the linear weights/parameters of Beta
		#	input_offset: (scalar), the GLM's offset for the input
		#	output_gain: (scalar), the GLM's gain/scalar for its output (after the link function)
		#	output_offset: (scalar), the GLM's offset for its output (after the link function)

		weights = self.model.get_weights()

		linear_weights = weights[0]
		input_offset = weights[1][0]
		output_gain = weights[6][0,0]
		output_offset = weights[7][0]

		return linear_weights, input_offset, output_gain, output_offset


	def get_predicted_output(self, X):
		# Predicts output data y from a given input data X
		#
		# INPUT:
		#	X: (num_input_vars, num_samples), input data (i.e., features)
		#
		# OUTPUT:
		#	y_hat: (num_samples,), predicted output data (i.e., targets)

		return np.squeeze(self.model.predict(X.T))


	def compute_R2(self, y_hat, y_test, flag_pearson=False):
		# Computes prediction performance R2 between model predictions y_hat and real data y_test
		#
		# INPUT:
		#   y_hat: (num_test_samples,), predicted output data from GLM; e.g., y_hat = GLM.get_predicted_output(X_test)
		#   y_test: (num_test_samples,), output data held out from training
		#   flag_pearson: (True or False), denotes if computing the coefficient of determination CoD (False) or
		#	      squaring Pearson's correlation rho^2 (True).
		#	      difference: CoD takes into account if y_hat's mean, std, and sign are different from y_test; rho^2 does not.
		#
		# OUTPUT:
		#   R2: (scalar), prediction performance between y_hat and y_true
		#	R2 --> 1 indicates the model has perfect prediction
		#	R2 --> 0 indicates the model is a mismatch and/or features do not relate to y
		#	R2 < 0 indicates truly poor prediction; this occurs when y_hat's mean or std dev is different from y_test,
		#	     y_hat and y_test are negatively correlated, etc.

		# error checking
		y_hat = np.squeeze(y_hat); y_test = np.squeeze(y_test)
		if y_hat.ndim != 1 or y_test.ndim != 1:
			raise ValueError('y_hat and y_test need to be one-dimensional vectors.')
		elif y_hat.size != y_test.size:
			raise ValueError('y_hat and y_test need to have the same number of samples.')
			
		if flag_pearson == False:
			sumXY = np.sum((y_hat - y_test)**2)
			sumYY = np.sum((y_test - np.mean(y_test))**2)

			R2 = 1. - sumXY / sumYY
		else:
			R2 = np.corrcoef(y_hat, y_test)[0,1]**2
		return R2


	def save_model(self, filetag='model', save_folder=None):
		# saves GLM model to specified filename + folder
		#
		# INPUT:
		#	filetag: (string), filename with no ending suffix
		#	save_folder: (string), folder path in which to save the file
		#		if None, saves to current folder
		# OUTPUT:
		#	None.

		if save_folder == None:
			save_folder = './'

		self.model.save(save_folder + filetag + '.h5')


	def load_model(self, filetag, load_folder=None):
		# saves GLM model to specified filename + folder
		#
		# INPUT:
		#	filetag: (string), filename with no ending suffix
		#	save_folder: (string), folder path in which to load the file
		#		if None, saves to current folder
		# OUTPUT:
		#	None.

		if load_folder == None:
			load_folder = './'

		self.model = load_model(load_folder + filetag + '.h5')
