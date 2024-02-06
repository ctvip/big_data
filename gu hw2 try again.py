"""
Start file for hw2pr3 of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

First, fill in the the code of step 0 in the main driver to load the data, then
please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""


#########################################
#			 Helper Functions	    	#
#########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################
#		 Part C			#
#########################
def linreg(X, y, reg=0.0):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		reg is the parameter for regularization.

		Return the optimal weight matrix.
	'''
	# Hint: Find the numerical solution for part c
	# Use np.eye to create identity matrix
	# Use np.linalg.solve to solve W_opt

	# YOUR CODE GOES BELOW
	eye = np.eye(X.shape[1])
	eye[0, 0] = 0 # don't regularize the bias term
	W_opt = np.linalg.solve(X.T @ X + reg * eye, X.T @ y)
	return W_opt

def predict(W, X):
	'''
		W is a weight matrix with bias.
		X is the data with dimension m x (n + 1).

		Return the predicted label, y_pred.
	'''
	return X @ W

def find_RMSE(W, X, y):
	'''
		W is the weight matrix with bias.
		X is the data with dimension m x (n + 1).
		y is label with dimension m x 1.

		Return the root mean-squared error.
	'''
	# YOUR CODE GOES BELOW
	y_pred = predict(W, X)
	diff = y - y_pred
	m = X.shape[0]
	MSE = np.linalg.norm(diff, 2) ** 2 / m
	return np.sqrt(MSE)

def RMSE_vs_lambda(X_train, y_train, X_val, y_val):
    # Step 1: Generate a list of lambda values
    reg_list = np.random.uniform(-1, 5, 150)  # for example, 100 values between 10^-4 and 10^2

    RMSE_list = []
    W_list = []

    # Step 2 & 3: Compute W_opt and RMSE for each lambda
    for reg in reg_list:
        W_opt = linreg(X_train, y_train, reg)
        W_list.append(W_opt)
        RMSE = find_RMSE(W_opt, X_val, y_val)
        RMSE_list.append(RMSE)

    # Step 4: Plot RMSE vs lambda
    plt.scatter(reg_list, RMSE_list, color='red')
    # plt.xscale('log')
    plt.title('RMSE vs lambda')
    plt.xlabel('lambda')
    plt.ylabel('RMSE')
    plt.savefig('RMSE_vs_lambda.png')
    plt.show()
    print('==> Plotting completed.')

    # Step 5: Find the optimal lambda
    reg_opt = reg_list[np.argmin(RMSE_list)]
    return reg_opt


def norm_vs_lambda(X_train, y_train, X_val, y_val):
	'''
		X is the data with dimension m x (n + 1).
		y is the label with dimension m x 1.

		Genearte a plot of norm of the weights vs lambda.
	'''
	# You may reuse the code for RMSE_vs_lambda
	# to generate the list of weights and regularization parameters
	# YOUR CODE GOES BELOW
	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = lb) for lb in reg_list]

	# Calculate the norm of each weight
	# YOUR CODE GOES BELOW
	norm_list = [np.linalg.norm(W, 2) for W in W_list]

	# Plot norm vs lambda
	norm_vs_lambda_plot, = plt.plot(reg_list, norm_list)
	plt.setp(norm_vs_lambda_plot, color = 'blue')
	plt.title('norm vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('norm')
	plt.savefig('norm_vs_lambda.png', format = 'png')
	plt.show()
	print('==> Plotting completed.')



#########################
#		 Part D			#
#########################


def linreg_no_bias(X, y, reg=0.0):
    """
    Perform linear regression without the bias term.

    Parameters:
    X -- the data matrix with dimension m x n (without the bias column)
    y -- the label of the data with dimension m x 1
    reg -- the parameter for regularization

    Returns:
    b_opt -- the optimal bias
    W_opt -- the optimal weight matrix
    """
    t_start = time.time()

    # Step 1: Regularize and solve for W_opt
    m, n = X.shape
    # Do not regularize the bias term, which is not included in X
    reg_matrix = reg * np.eye(n)
    # Solve for the optimal weights (excluding the bias)
    W_opt = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)

    # Step 2: Compute the bias term b_opt
    # The bias is the mean of y minus the mean of the predictions from the non-biased model
    y_pred = X @ W_opt
    b_opt = np.mean(y - y_pred)

    # Benchmark report
    t_end = time.time()
    print('--Time elapsed for training: {t:4.2f} seconds'.format(t=t_end - t_start))

    return b_opt, W_opt

#########################
#		 Part E			#
#########################


def grad_descent(X_train, y_train, X_val, y_val, reg = 0.0, \
	lr_W = 2.5e-12, lr_b = 0.2, max_iter = 150, eps = 1e-6, print_freq = 25):
	'''
		X is matrix with dimension m x n.
		y is label with dimension m x 1.
		reg is the parameter for regularization.
		lr_W is the learning rate for weights.
		lr_b is the learning rate for bias.
		max_iter is the maximum number of iterations.
		eps is the threshold of the norm for the gradients.
		print_freq is the frequency of printing the report.

		Return the optimal weight and bias by gradient descent.
	'''
	m_train, n = X_train.shape
	m_val = X_val.shape[0]
	# initialize the weights and bias and their corresponding gradients
	# YOUR CODE GOES BELOW
	W = np.zeros((n, 1))
	b = 0.
	W_grad = np.ones_like(W)
	b_grad = 1.

	obj_train = []
	obj_val = []
	print('==> Running gradient descent...')
	iter_num = 0
	t_start = time.time()
	# Running the gradient descent algorithm
	# First, calculate the training rmse and validation rmse at each iteration
	# Append these values to obj_train and obj_val respectively
	# Then, calculate the gradient for W and b as W_grad and b_grad
	# Upgrade W and b
	# Keep iterating while the number of iterations is less than the maximum
	# and the gradient is larger than the threshold
	# YOUR CODE GOES BELOW
	while np.linalg.norm(W_grad) > eps and np.linalg.norm(b_grad) > eps \
	and iter_num < max_iter:
		# calculate norms
		train_rmse = np.sqrt(np.linalg.norm((X_train @ W).reshape((-1, 1)) \
			+ b - y_train) ** 2 / m_train)
		obj_train.append(train_rmse)
		val_rmse = np.sqrt(np.linalg.norm((X_val @ W).reshape((-1, 1)) \
			+ b - y_val) ** 2 / m_val)
		obj_val.append(val_rmse)
		# calculate gradient
		W_grad = ((X_train.T @ X_train + reg * np.eye(n)) @ W \
			+ X_train.T @ (b - y_train)) / m_train
		b_grad = (sum(X_train @ W) - sum(y_train) + b * m_train) / m_train
		# update weights and bias
		W -= lr_W * W_grad
		b -= lr_b * b_grad
		# print statements
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration{} - training rmse {: 4.4f} - gradient norm {: 4.4E}'.format(\
				iter_num + 1, train_rmse, np.linalg.norm(W_grad)))
		iter_num += 1

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} seconds'.format(\
			t = t_end - t_start))
	# generate convergence plot
	train_rmse_plot, = plt.plot(range(iter_num), obj_train)
	plt.setp(train_rmse_plot, color = 'red')
	val_rmse_plot, = plt.plot(range(iter_num), obj_val)
	plt.setp(val_rmse_plot, color = 'green')
	plt.legend((train_rmse_plot, val_rmse_plot), \
		('Training RMSE', 'Validation RMSE'), loc = 'best')
	plt.title('RMSE vs iteration')
	plt.xlabel('iteration')
	plt.ylabel('RMSE')
	plt.savefig('convergence.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	return b, W



###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')

	# Read data
	df = pd.read_csv('https://math189sp19.github.io/data/online_news_popularity.csv', \
		sep=', ', engine='python')

	# split the data frame by type: training, validation, and test
	train_pct = 2.0 / 3
	val_pct = 5.0 / 6

	df['type'] = ''
	df.loc[:int(train_pct * len(df)), 'type'] = 'train'
	df.loc[int(train_pct * len(df)) : int(val_pct * len(df)), 'type'] = 'val'
	df.loc[int(val_pct * len(df)):, 'type'] = 'test'


	# extracting columns into training, validation, and test data
	X_train = np.array(df[df.type == 'train'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_train = np.array(np.log(df[df.type == 'train'].shares)).reshape((-1, 1))

	X_val = np.array(df[df.type == 'val'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_val = np.array(np.log(df[df.type == 'val'].shares)).reshape((-1, 1))

	X_test = np.array(df[df.type == 'test'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_test = np.array(np.log(df[df.type == 'test'].shares)).reshape((-1, 1))
      


	# TODO: Stack a column of ones to the feature data, X_train, X_val and X_test

	# HINT:
	# 	1) Use np.ones / np.ones_like to create a column of ones
	#	2) Use np.hstack to stack the column to the matrix
	"*** YOUR CODE HERE ***"
	

	"*** END YOUR CODE HERE ***"

	# Convert data to matrix
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_val = np.matrix(X_val)
	y_val = np.matrix(y_val)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)



	# PART C
	# =============STEP 1: RMSE vs lambda=================
	# NOTE: Fill in code in linreg, findRMSE, and RMSE_vs_lambda for this step

	print('==> Step 1: RMSE vs lambda...')

	# find the optimal regularization parameter
	reg_opt = RMSE_vs_lambda(X_train, y_train, X_val, y_val)
	print('==> The optimal regularization parameter is {reg: 4.4f}.'.format(\
		reg=reg_opt))

	# Find the optimal weights and bias for future use in step 3
	W_with_b_1 = linreg(X_train, y_train, reg=reg_opt)
	b_opt_1 = W_with_b_1[0]
	W_opt_1 = W_with_b_1[1: ]

	# Report the RMSE with the found optimal weights on validation set
	val_RMSE = find_RMSE(W_with_b_1, X_val, y_val)
	print('==> The RMSE on the validation set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=val_RMSE))

	# Report the RMSE with the found optimal weights on test set
	test_RMSE = find_RMSE(W_with_b_1, X_test, y_test)
	print('==> The RMSE on the test set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=test_RMSE))



	# =============STEP 2: Norm vs lambda=================
	# NOTE: Fill in code in norm_vs_lambda for this step

	print('\n==> Step 2: Norm vs lambda...')
	norm_vs_lambda(X_train, y_train, X_val, y_val)



	# PART D
	# =============STEP 3: Linear regression without bias=================
	# NOTE: Fill in code in linreg_no_bias for this step

	# From here on, we will strip the columns of ones for all data
	X_train = X_train[:, 1:]
	X_val = X_val[:, 1:]
	X_test = X_test[:, 1:]

	# Compare the result with the one from step 1
	# The difference in norm should be a small scalar (i.e, 1e-10)
	print('\n==> Step 3: Linear regression without bias...')
	b_opt_2, W_opt_2 = linreg_no_bias(X_train, y_train, reg=reg_opt)

	# difference in bias
	diff_bias = np.linalg.norm(b_opt_2 - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_opt_2 -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))



	# PART E
	# =============STEP 4: Gradient descent=================
	# NOTE: Fill in code in grad_descent for this step

	print('\n==> Step 4: Gradient descent')
	b_gd, W_gd = grad_descent(X_train, y_train, X_val, y_val, reg=reg_opt)

	# Compare the result from the one from step 1
	# Difference in bias
	diff_bias = np.linalg.norm(b_gd - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_gd -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))
