import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from Conv_NN_layer_utils import *

import pdb
class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  {conv- [batch norm] - relu - pool}xN - { affine - [batch norm] - relu - [dropout] }xM - affine - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), hidden_dims_CNN = ((32, 5, 1, 1), (2, 2, 2)),
               hidden_dims_FC = ((1024), (0.5)), num_classes=10, weight_scale=1e-3, 
               reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim (C, H, W): Tuple giving size of input data
    - hidden_dims_CNN ((F, filter_size, stride, pad), (pooling_stride, pooling_height, pooling_width)) * N
    - hidden_dims_FC (fc_num, dropout_ratio) * M
    - num_classes: Number of output classes
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.fix_params = {}
    self.reg = reg
    self.dtype = dtype
    
    C_input, H_input, W_input = input_dim
    pre_C = C_input 
    pre_H = H_input
    pre_W = W_input
    
    num_CNN = len(hidden_dims_CNN)
    num_FC = len(hidden_dims_FC)

    for i in range(0, num_CNN):
      W_name = "W" + str(i)
      b_name = "b" + str(i)
      conv_param_name = "conv_param" + str(i)
      gamma_name = "gamma" + str(i)
      beta_name = "beta" + str(i)
      bn_param_name = "bn_param" + str(i)
      pool_param_name = "pool_param" + str(i)

      if num_CNN == 1:
        num_filters, filter_size, stride, pad = hidden_dims_CNN[0] # (F, filter_size, stride, pad)
        pool_stride, pool_height, pool_width = hidden_dims_CNN[1] # (pooling_stride, pooling_size)
      else:
        num_filters, filter_size, stride, pad = hidden_dims_CNN[i][0] # (F, filter_size, stride, pad)
        pool_stride, pool_height, pool_width = hidden_dims_CNN[i][1] # (pooling_stride, pooling_size)
      
      if weight_scale == -1:
        self.params[W_name] = np.random.randn(num_filters, pre_C, filter_size, filter_size) / np.sqrt(filter_size * filter_size * pre_C)
      else: 
        self.params[W_name] = np.random.randn(num_filters, pre_C, filter_size, filter_size) * weight_scale
      self.params[b_name] = np.zeros(num_filters)
      self.fix_params[conv_param_name] = {'stride': stride, 'pad': pad}
      
      self.params[gamma_name] = np.random.randn(num_filters)
      self.params[beta_name] = np.random.randn(num_filters)
      self.fix_params[bn_param_name] = {'mode': 'train'}

      self.fix_params[pool_param_name] = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}
      
      pre_H, pre_W = cnn_out_shape(pre_H, pre_W, filter_size, filter_size, stride, pad)
      pre_C = num_filters 
      pre_H, pre_W = pool_out_shape(pre_H, pre_W, pool_height, pool_width, pool_stride)

    pre_fc_dim = pre_H * pre_W * pre_C

    for i in range(0, num_FC):
      W_name = "W" + str(i + num_CNN)
      b_name = "b" + str(i + num_CNN)
      gamma_name = "gamma" + str(i + num_CNN)
      beta_name = "beta" + str(i + num_CNN)
      bn_param_name = "bn_param" + str(i + num_CNN)
      drop_name = "drop_ratio" + str(i + num_CNN)
      
      if num_FC == 1 :
        fc_num = hidden_dims_FC[0]
        drop_ratio = hidden_dims_FC[1]
      else:
        fc_num = hidden_dims_FC[i][0]
        drop_ratio = hidden_dims_FC[i][1]

      if weight_scale == -1:
        self.params[W_name] = np.random.randn(pre_fc_dim, fc_num) / np.sqrt(pre_fc_dim)
      else:
        self.params[W_name] = np.random.randn(pre_fc_dim, fc_num) * weight_scale
      self.params[b_name] = np.zeros(fc_num)

      self.params[gamma_name] = np.random.randn(fc_num)
      self.params[beta_name] = np.random.randn(fc_num)
      self.fix_params[bn_param_name] = {'mode': 'train'}

      self.fix_params[drop_name] = {'mode': 'train', 'p': drop_ratio}

      pre_fc_dim = fc_num

    total_layer = num_CNN + num_FC
    W_name = "W" + str(total_layer)
    b_name = "b" + str(total_layer)
    if weight_scale == -1:
      self.params[W_name] = np.random.randn(pre_fc_dim, num_classes) / np.sqrt(pre_fc_dim)
    else:
      self.params[W_name] = np.random.randn(pre_fc_dim, num_classes) * weight_scale
    self.params[b_name] = np.zeros(num_classes)


    self.num_CNN = num_CNN
    self.num_FC = num_FC
    self.total_layer = num_CNN + num_FC

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    num_FC = self.num_FC
    num_CNN = self.num_CNN
    total_layer = self.num_FC + self.num_CNN
    
    cache = {}
    pre_layer_output = X
    for i in range(0, num_CNN):
      W_name = "W" + str(i)
      b_name = "b" + str(i)
      conv_param_name = "conv_param" + str(i)
      gamma_name = "gamma" + str(i)
      beta_name = "beta" + str(i)
      bn_param_name = "bn_param" + str(i)
      pool_param_name = "pool_param" + str(i)

      w = self.params[W_name]
      b = self.params[b_name]
      conv_param = self.fix_params[conv_param_name]
      gamma = self.params[gamma_name]
      beta = self.params[beta_name]
      bn_param = self.fix_params[bn_param_name]
      pool_param = self.fix_params[pool_param_name]
      
      pre_layer_output, cache_layer = cnn_batch_relu_pool_forward(pre_layer_output, 
                                                                  w, b, conv_param, 
                                                                  gamma, beta, bn_param, 
                                                                  pool_param)
      cache[i] = cache_layer
    
    for i in range(0, num_FC):
      W_name = "W" + str(i + num_CNN)
      b_name = "b" + str(i + num_CNN)
      gamma_name = "gamma" + str(i + num_CNN)
      beta_name = "beta" + str(i + num_CNN)
      bn_param_name = "bn_param" + str(i + num_CNN)
      drop_name = "drop_ratio" + str(i + num_CNN)

      w = self.params[W_name]
      b = self.params[b_name]
      gamma = self.params[gamma_name]
      beta = self.params[beta_name]
      bn_param = self.fix_params[bn_param_name]
      dropout_param = self.fix_params[drop_name]

      pre_layer_output, cache_layer = affine_batch_relu_drop_forward(pre_layer_output, 
                                                                    w, b, 
                                                                    gamma, beta, bn_param, 
                                                                    dropout_param)
      cache[i + num_CNN] = cache_layer
    
    W_name = "W" + str(total_layer)
    b_name = "b" + str(total_layer)
    w = self.params[W_name]
    b = self.params[b_name]
    scores, cache[total_layer] = affine_forward(pre_layer_output, w, b)
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    loss, dUpLayer = softmax_loss(scores, y)
    loss = loss + 0.5 * self.reg * np.sum(w**2)
    
    cache_layer = cache[total_layer]
    dUpLayer, grads[W_name], grads[b_name] = affine_backward(dUpLayer, cache_layer)
    grads[W_name] = grads[W_name] + self.reg * self.params[W_name]

    for i in range(0, num_FC):
      layer_index = num_FC + num_CNN -1 - i
      W_name = "W" + str(layer_index)
      b_name = "b" + str(layer_index)
      gamma_name = "gamma" + str(layer_index)
      beta_name = "beta" + str(layer_index)

      cache_layer = cache[layer_index]
      dUpLayer, grads[W_name], grads[b_name], grads[gamma_name], grads[beta_name] = affine_batch_relu_drop_backward(dUpLayer, cache_layer)

      loss = loss + 0.5 * self.reg * np.sum(self.params[W_name]**2)
      grads[W_name] = grads[W_name] + self.reg * self.params[W_name]
      grads[gamma_name] = grads[gamma_name] + self.reg * self.params[gamma_name]

    for i in range(0, num_CNN):

      layer_index = num_CNN -1 - i

      W_name = "W" + str(layer_index)
      b_name = "b" + str(layer_index)
      conv_param_name = "conv_param" + str(layer_index)
      gamma_name = "gamma" + str(layer_index)
      beta_name = "beta" + str(layer_index)

      cache_layer = cache[layer_index]
      dUpLayer, grads[W_name], grads[b_name], grads[gamma_name], grads[beta_name] = cnn_batch_relu_pool_backward(dUpLayer, cache_layer)

      loss = loss + 0.5 * self.reg * np.sum(self.params[W_name]**2)
      grads[W_name] = grads[W_name] + self.reg * self.params[W_name]
      grads[gamma_name] = grads[gamma_name] + self.reg * self.params[gamma_name]
      
    return loss, grads




class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn( num_filters * (0.5 * H) * (0.5 * W), hidden_dim) * weight_scale # * sqrt(2.0/n)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale # * sqrt(2.0/n)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cnn_out, cnn_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    hidden_out, hidden_cache = affine_relu_forward(cnn_out, W2, b2)
    scores, scores_cache = affine_forward(hidden_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # Compute loss and gradients
    loss, dscores =  softmax_loss(scores, y)
    dhidden, grads['W3'], grads['b3'] = affine_backward(dscores, scores_cache)
    dcnn, grads['W2'], grads['b2'] = affine_relu_backward(dhidden, hidden_cache)
    dX, grads['W1'], grads['b1'] = conv_relu_pool_backward(dcnn, cnn_cache)

    # Regularization
    loss = loss + 0.5*self.reg*np.sum(self.params['W3']**2)
    loss = loss + 0.5*self.reg*np.sum(self.params['W2']**2)
    loss = loss + 0.5*self.reg*np.sum(self.params['W1']**2)
    grads['W3'] = grads['W3'] + self.reg * self.params['W3']
    grads['W2'] = grads['W2'] + self.reg * self.params['W2']
    grads['W1'] = grads['W1'] + self.reg * self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
