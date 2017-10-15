import pdb
import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores_stable = scores - np.max(scores)
    correct_class_score = scores[y[i]]
    exp_val = np.exp(scores_stable)
    #pdb.set_trace()
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] = dW[:,j] - X[i] + (exp_val[j]/np.sum(exp_val)) * X[i]
      else:
        dW[:,j] = dW[:,j] + (exp_val[j]/np.sum(exp_val)) * X[i]
      loss += np.exp(correct_class_score - np.max(scores))/np.sum(np.exp(scores - np.max(scores)))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  ### AG ###
  dW /= float(num_train)
  ###
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_naive_online_sol(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  X = X.T
  W = W.T
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_class = dW.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  loss = 0.0
  for i in xrange(num_train):
    X_i =  X[:,i]
    score_i = W.dot(X_i)
    stability = -score_i.max()
    exp_score_i = np.exp(score_i+stability)
    exp_score_total_i = np.sum(exp_score_i , axis = 0)
    for j in xrange(num_class):
      if j == y[i]:
        dW[j,:] += -X_i.T + (exp_score_i[j] / exp_score_total_i) * X_i.T
      else:
        dW[j,:] += (exp_score_i[j] / exp_score_total_i) * X_i.T
    numerator = np.exp(score_i[y[i]]+stability)
    denom = np.sum(np.exp(score_i+stability))
    if numerator == 0:
      pdb.set_trace()
    loss += -np.log(numerator / float(denom))
    

  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW.T

def softmax_loss_naive_test(W, X, y, reg):
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
  W = W.T
  X = X.T
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:,i])
    scores_stable = scores - np.max(scores)
    correct_class_score = scores[y[i]]
    exp_val = np.exp(scores_stable)
    #pdb.set_trace()
    for j in xrange(num_classes):
      if j == y[i]:
        dW[j,:] = dW[j,:] - X[:,i].T + (exp_val[j]/np.sum(exp_val)) * X[:,i].T
      else:
        dW[j,:] = dW[j,:] + (exp_val[j]/np.sum(exp_val)) * X[:,i].T
    loss += -np.log(exp_val[y[i]]/np.sum(exp_val))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= float(num_train)
  ### AG ###
  dW /= float(num_train)
  ###
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW.T

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

