import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def cnn_out_shape(H_X, W_X, HH, WW, stride, pad):
  H_out = 1 + np.floor((H_X + 2 * pad - HH) / stride)
  W_out = 1 + np.floor((W_X + 2 * pad - WW) / stride)
  
  return H_out, W_out

def pool_out_shape(H_X, W_X, pool_height, pool_width, stride):
  H_out = 1 + np.floor((H_X - pool_height) / stride)
  W_out = 1 + np.floor((W_X - pool_width) / stride)

  return H_out, W_out

def cnn_batch_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):
  """
  Convenience layer that performs {conv- [batch norm] - relu - pool}
  Input:
  - x: Input data of shape (N, C, H, W)
  
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  - gamma: Scale parameter, of shape (F,)
  - beta: Shift parameter, of shape (F,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions
  
  Returns a tuple of:
  - out: output data
  - cache: (conv_cache, bn_cache, relu_cache, pool_cache)
  """

  cnn_out, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn_out, bn_cache = spatial_batchnorm_forward(cnn_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)
  out, pool_cache = max_pool_forward_fast(relu_out, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  
  return out, cache

def cnn_batch_relu_pool_backward(dout, cache):
  """
  Backward pass for the  {conv- [batch norm] - relu - pool}
  """
  (conv_cache, bn_cache, relu_cache, pool_cache) = cache
  dpool_out = max_pool_backward_fast(dout, pool_cache)
  drelu_out = relu_backward(dpool_out, relu_cache)
  dbn_out, dgamma, dbeta = spatial_batchnorm_backward(drelu_out, bn_cache)
  dx, dw, db = conv_backward_fast(dbn_out, conv_cache)
  
  return dx, dw, db, dgamma, dbeta

def affine_batch_relu_drop_forward(x, w, b, gamma, beta, bn_param, dropout_param):
  """
  Convenience layer that perorms an { affine - [batch norm] - relu - [dropout] }

  Inputs:
  - x: Data of shape (N, D)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  - gamma: Scale parameter of shape (M,)
  - beta: Shift paremeter of shape (M,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (M,) giving running mean of features
    - running_var Array of shape (M,) giving running variance of features

  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Returns a tuple of:
  - out: Output from the ReLU/Dropout
  - cache: (fc_cache, bn_cache, relu_cache, drop_cache)
  """
  fc_out, fc_cache = affine_forward(x, w, b)
  bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)
  out, drop_cache = dropout_forward(relu_out, dropout_param)
  cache = (fc_cache, bn_cache, relu_cache, drop_cache)

  return out, cache

def affine_batch_relu_drop_backward(dout, cache):
  """
  Backward pass for the { affine - [batch norm] - relu - [dropout] } 
  """
  (fc_cache, bn_cache, relu_cache, drop_cache) = cache
  ddrop_out = dropout_backward(dout, drop_cache)
  drelu_out = relu_backward(ddrop_out, relu_cache)
  dbn_out, dgamma, dbeta =  batchnorm_backward_alt(drelu_out, bn_cache)
  dx, dw, db = affine_backward(dbn_out, fc_cache)

  return dx, dw, db, dgamma, dbeta

  
