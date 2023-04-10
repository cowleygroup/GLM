
import numpy as np
import class_model_GLM



### HELPER FUNCTIONS

def simulate_X_and_y(num_input_vars, link_function='linear', epsilon_sigma=0.25, num_train_samples=500, num_test_samples=100):
	# Simulates relationship between input data X and output data y, following a GLM.
	#
	# INPUT:
	#	num_input_vars: (int), number of input/feature variables of X
	#	link_function: ('linear', 'relu', 'sigmoid'), pointwise nonlinear function between Beta^T * X and y
	#	epsilon_sigma: (float), std. dev. of the epsilon noise added to y (after the link function)
	#	num_train_samples: (int), number of samples generated for training
	#	num_test_samples: (int), number of samples generated for testing
	#
	# OUTPUT:
	#	X_train: (num_input_vars, num_train_samples), input data for training (i.e., features)
	#	y_train: (num_train_samples,), output data for training (i.e., target)
	#	X_test: (num_input_vars, num_test_samples), input data for training (i.e., features)
	#	y_test: (num_test_samples,), output data for testing (i.e., target)	

	# generate input data X
	X_train = np.random.normal(loc=0., scale=1., size=(num_input_vars,num_train_samples))
	X_test = np.random.normal(loc=0., scale=1., size=(num_input_vars,num_test_samples))

	# generate ground truth weights
	Beta_star = np.random.normal(loc=0., scale=1., size=(num_input_vars,))

	# generate output data y
	y_train = np.sum(Beta_star[:,np.newaxis] * X_train,axis=0)
	y_test = np.sum(Beta_star[:,np.newaxis] * X_test,axis=0)

	if link_function == 'relu':
		y_train = np.clip(y_train, a_min=0, a_max=None)
		y_test = np.clip(y_test, a_min=0, a_max=None)
	elif link_function == 'sigmoid':
		y_train = np.exp(-y_train) / (1. + np.exp(-y_train))
		y_test = np.exp(-y_test) / (1. + np.exp(-y_test))

	output_offset = 5.
	epsilon_noise = np.random.normal(loc=output_offset, scale=epsilon_sigma, size=(num_train_samples,))
	y_train = y_train + epsilon_noise
	epsilon_noise = np.random.normal(loc=output_offset, scale=epsilon_sigma, size=(num_test_samples,))
	y_test = y_test + epsilon_noise

	return X_train, np.squeeze(y_train), X_test, np.squeeze(y_test)



### MAIN SCRIPT
#
# Generates fake data X and y from generalized linear model (GLM) and uses this data to train a GLM.
# Useful for practicing training GLMs and gaining intuition.

num_input_vars = 20
link_function = 'linear'

X_train, y_train, X_test, y_test = simulate_X_and_y(num_input_vars, link_function=link_function)

mean_output = np.mean(y_train)
stddev_output = np.std(y_train)

GLM = class_model_GLM.GLMModelClass(num_input_vars, link_function=link_function)

# train model
for ipass in range(10):
  print('training pass {:d}'.format(ipass))
  GLM.train_model(X_train, y_train)

# test model
y_hat = GLM.get_predicted_output(X_test)
R2 = GLM.compute_R2(y_hat, y_test)

print('R2 = {:f}'.format(R2))






