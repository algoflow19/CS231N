import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores = scores - np.amax(scores)
      exp_scores = np.exp(scores)
      denom = np.sum(exp_scores)
      numer = exp_scores[y[i]]
      loss -= np.log(numer / denom)
      grad = exp_scores / denom
      grad[y[i]] -= 1
      dW += X[i].reshape((-1, 1)).dot(grad.reshape((1, -1))) 

  
  loss = loss / float(num_train) + reg * np.sum(W * W)
  dW = dW / float(num_train) + reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores - np.amax(scores, axis=1).reshape((-1, 1))
  exp_scores = np.exp(scores)
  denoms = np.sum(exp_scores, axis=1)
  y_hat = exp_scores / denoms.reshape((-1, 1))
  loss = np.sum(-np.log(y_hat[np.arange(num_train), y])) / float(num_train) + reg * np.sum(W * W)
  y_hat[np.arange(num_train), y] -= 1
  dW = X.T.dot(y_hat) / float(num_train) + reg * 2 * W 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

