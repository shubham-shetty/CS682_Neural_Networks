import numpy as np
from random import shuffle
import math


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #exp_matrix = np.zeros_like()
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    x_i = X[i]
    y_i = y[i]
    all_scores = np.dot(x_i, W)
    correct_class_score = all_scores[y_i]
    exp_list = np.exp(all_scores)
    exp_sum = np.sum(exp_list)
    exp_corr_class =  exp_list[y_i]
    loss += - np.log(exp_corr_class/exp_sum)
    dW[:,y_i] += (-1) * ((exp_sum-exp_corr_class)/exp_sum) * x_i
    for j in range(num_classes):
      if j == y_i:
        continue
      dW[:, j] += (exp_list[j] / exp_sum) * x_i
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * 2*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  all_scores = np.dot(X, W) #NxC
  correct_class_scores = all_scores[np.arange(len(all_scores)),y] #N
  exp_list = np.exp(all_scores) #NxC
  exp_sum = np.sum(exp_list, axis = 1) #N
  exp_corr_class = np.exp(correct_class_scores) #N
  loss += np.sum((-1)*np.log(exp_corr_class/exp_sum)) #1x1
  
  sm = exp_list / exp_sum.reshape(num_train,1)#NxC
  sm[range(num_train),y] = (-1) * ((exp_sum - exp_corr_class) / exp_sum)
  dW = np.dot(X.T, sm) #DxC

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * 2*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

