"""
Start file for hw4pr2 part (b) of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

First, please COMMENT OUT any steps other than step 0 in main driver before
you finish the corresponding functions for that step. Otherwise, you won't be
able to run the program because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

2. Remember to comment out the TODO comment after you finish each part.
"""


#########################################
#			 Helper Functions	    	#
#########################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time



def softmax(z):
    """Compute softmax values for each set of scores in z."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    return e_z / e_z.sum(axis=1, keepdims=True)
def NLL(X, y_OH, W, reg=0.0):
    # Compute class scores
    scores = np.dot(X, W)
    probs = softmax(scores)

    # Convert one-hot encoded labels to class indices
    class_indices = np.argmax(y_OH, axis=1)

    # Calculate negative log likelihood
    nll = -np.sum(np.log(probs[np.arange(len(y_OH)), class_indices]))

    # Add regularization term
    nll += reg / 2.0 * np.sum(W**2)

    return nll / X.shape[0]

def grad_softmax(X, y_OH, W, reg=0.0):
    m = X.shape[0]

    # Compute class scores
    scores = np.dot(X, W)

    # Apply softmax to compute probabilities
    probs = softmax(scores)

    # Compute the gradient
    grad = np.dot(X.T, (probs - y_OH)) / m

    # Apply L2 regularization
    grad += reg * W / m

    return grad


def predict(X, W):
    """Return the predicted labels y_pred using X and W."""
    # Compute the class scores
    scores = np.dot(X, W)

    # Apply softmax to get probabilities
    probs = softmax(scores)

    # Use np.argmax to get the predicted label for each example
    y_pred = np.argmax(probs, axis=1).reshape(-1, 1)

    return y_pred


def get_accuracy(y_pred, y):
    diff = (y_pred == y).astype(int)
    accu = 1.0 * diff.sum() / len(y)
    return accu



def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, max_iter=500, print_freq=20):
    m, n = X.shape
    k = y.shape[1]  # Correct way to determine 'k' for one-hot encoded labels
    nll_list = []

    # Initialize the weight matrix W
    W = np.zeros((n, k))
    W_grad = np.ones((n, k))

    print('\n==> Running gradient descent...')

    t_start = time.time()

    iter_num = 0
    while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
        # Calculate the negative log likelihood
        nll = NLL(X, y, W, reg)

        # Check for NaN
        if np.isnan(nll):
            print("NaN encountered in NLL computation.")
            break

        nll_list.append(nll)

        # Calculate the gradient for W
        W_grad = grad_softmax(X, y, W, reg)

        # Update W
        W -= lr * W_grad

        # Print statements for debugging
        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(iter_num + 1, nll))

        iter_num += 1

    t_end = time.time()
    print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(t=t_end - t_start))

    return W, nll_list


def accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list):
    accu_list = []

    for reg in lambda_list:
        # Run gradient descent
        W_opt, _ = grad_descent(X_train, y_train_OH, reg=reg)

        # Predict labels
        y_pred = predict(X_test, W_opt)

        # Calculate accuracy
        accuracy = get_accuracy(y_pred, y_test)
        accu_list.append(accuracy)

        print('-- Accuracy is {:2.4f} for lambda = {:2.2f}'.format(accuracy, reg))

    # Plot accuracy vs lambda
    print('==> Printing accuracy vs lambda...')
    plt.style.use('ggplot')
    plt.plot(lambda_list, accu_list)
    plt.title('Accuracy versus Lambda in Softmax Regression')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.savefig('hw4pr2b_lva.png', format='png')
    plt.close()
    print('==> Plotting completed.')

    # Find the optimal lambda that maximizes accuracy
    reg_opt = lambda_list[np.argmax(accu_list)]

    return reg_opt





###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':


	# =============STEP 0: LOADING DATA=================
	# NOTE: The data is loaded using the code in p2_data.py. Please make sure
	#		you read the code in that file and understand how it works.
     

	# Load the data
	xtrain = np.load("data/X_train.npy", allow_pickle=True)
	ytrain = np.load("data/y_train.npy", allow_pickle=True).astype(int)
	xtest = np.load("data/X_test.npy", allow_pickle=True)
	ytest = np.load("data/y_test.npy", allow_pickle=True).astype(int)

# Normalize the features
	X_train = xtrain / 256
	X_test = xtest / 256

# One-hot encode the labels
	enc = OneHotEncoder(sparse=False, categories='auto')
	y_train_OH = enc.fit_transform(ytrain.reshape(-1, 1))
	y_test_OH = enc.transform(ytest.reshape(-1, 1))

# Add a column of ones to both training and testing data for the bias term
	X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
	X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Check if the training data is empty
	if X_train.size == 0 or y_train_OH.size == 0:
		print("Error: Training data is empty.")




	# =============STEP 1: Accuracy versus lambda=================
	# NOTE: Fill in the code in NLL, grad_softmax, predict and grad_descent.
	# 		Then, fill in predict and accuracy_vs_lambda

	print('\n\n==> Step 1: Finding optimal regularization parameter...')

	lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
	reg_opt = accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test_OH, lambda_list)

	print('\n-- Optimal regularization parameter is {:2.2f}'.format(reg_opt))





	# =============STEP 2: Convergence plot=================
	# NOTE: You DO NOT need to fill in any additional helper function for this
	# 		step to run. This step uses what you implemented for the previous
	#		step to plot.

	# run gradient descent to get the nll_list
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=reg_opt,\
	 	max_iter=1500, lr=2e-5, print_freq=100)

	print('\n==> Step 2: Plotting convergence plot...')

	# set up style for the plot
	plt.style.use('ggplot')

	# generate the convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	# add legend, title, etc and save the figure
	plt.title('Convergence Plot on Softmax Regression with $\lambda = {:2.2f}$'.format(reg_opt))
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2b_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')
