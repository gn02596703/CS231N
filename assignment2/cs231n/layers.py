import numpy as np
import pdb

def affine_forward(X, W, B):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  num_example = X.shape[0]
  X_Dim = len(X.shape) -1
  Feature_2D_len =1
  for i in range(1, X_Dim +1):
    Feature_2D_len = Feature_2D_len* X.shape[i] 
  X_2d = X.reshape(-1, Feature_2D_len).T
  W_t = W.T
  B_add = B.reshape(1, B.size)
  B_add = np.repeat(B_add, num_example, axis = 0)
  #pdb.set_trace()
  out = W_t.dot(X_2d) + B_add.T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (X, W, B)
  return out.T, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  X, W, B = cache
  dX, dW, dB = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  num_example = X.shape[0]
  X_Dim = len(X.shape) -1
  X_Dim_array = [num_example]
  Feature_2D_len =1
  for i in range(1, X_Dim +1):
    Feature_2D_len = Feature_2D_len* X.shape[i] 
    X_Dim_array.append(X.shape[i])
  X_2d = X.reshape(-1, Feature_2D_len).T
  W_t = W.T
  #pdb.set_trace()
  dX = dout.dot(W_t)
  dX = dX.reshape(X_Dim_array)
  dW = ((dout.T).dot(X_2d.T)).T
  dB = np.sum(dout, axis = 0).T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dX, dW, dB


def relu_forward(X):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.copy(X)
  index = out < 0
  out[index] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = X
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dX, X = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dX = np.copy(dout)
  index = X < 0
  dX[index] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dX


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0)
    sample_var = np.var(x, axis = 0)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    #X_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
    #out = gamma * X_norm + beta

    # Computation graph of whole work flow 
    num_train, num_feature = x.shape
    MX = np.mean(x, axis = 0)
    VarX = np.var(x, axis = 0)
    VarEps = VarX + eps
    SqVarEps = np.sqrt(VarEps)
    iSqVarEps = SqVarEps ** (-1)
    Xu = x - MX
    Xhat = Xu * iSqVarEps
    out = gamma * Xhat + beta

    cache = (x, gamma, beta, sample_mean, sample_var, bn_param)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    X_norm = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * X_norm + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

# Back Propagation of Batch Normalization
# http://cthorey.github.io./backpropagation/
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x, gamma, beta, sample_mean, sample_var, bn_param = cache
  eps = bn_param.get('eps', 1e-5)

  num_train, num_feature = x.shape
  MX = np.mean(x, axis = 0)
  VarX = np.var(x, axis = 0)
  VarEps = VarX + eps
  SqVarEps = np.sqrt(VarEps)
  iSqVarEps = SqVarEps ** (-1)
  Xu = x - MX
  Xhat = Xu * iSqVarEps
  out = gamma * Xhat + beta
  
  dy_dXhat = dout * gamma
  dXhat_dXu = dy_dXhat * iSqVarEps

  dXu_dx = dXhat_dXu # 1
  dXu_dMX = -1 * np.sum(dXhat_dXu, axis = 0)
  dMx_dx = dXu_dMX * np.ones((num_train, num_feature)) / np.float(num_train) #2

  dXhat_diSqVarEps = np.sum(dy_dXhat * Xu, axis = 0)
  diSqVarEps_dSqVarEps = -1. / (SqVarEps**(2)) * dXhat_diSqVarEps
  dSqVarEps_dVarEps = 0.5 / np.sqrt(VarEps) * diSqVarEps_dSqVarEps
  dVarEps_dVarX = dSqVarEps_dVarEps
  dVarX_dXu = 1. / np.float(num_train)* 2 * Xu * np.ones((num_train, num_feature)) * dVarEps_dVarX 
  
  dXu_dx_2 = dVarX_dXu # 1
  dXu_dMX_2 = (-1) * np.sum(dVarX_dXu, axis = 0)
  dMx_dx_2 = dXu_dMX_2 * np.ones((num_train, num_feature)) / np.float(num_train) #2
  
  dx = dXu_dx + dXu_dx_2 + dMx_dx + dMx_dx_2
  

  dgamma = np.sum(dout * Xhat, axis = 0)
  dbeta = np.sum(dout, axis = 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x, gamma, beta, sample_mean, sample_var, bn_param = cache
  N, D = x.shape
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  
  X_norm = (x - sample_mean) / np.sqrt(sample_var + eps)

  dx_norm = dout * gamma
  dsample_var = np.sum(dx_norm * (x - sample_mean) * (-0.5) * ((sample_var + eps)**(-1.5)) , axis = 0)
  dsample_mean = np.sum(dx_norm * (-1) / np.sqrt(sample_var + eps), axis = 0) + dsample_var * np.sum(-2 * (x - sample_mean), axis = 0) / np.float(N)
  
  dx = dx_norm / np.sqrt(sample_var + eps) + dsample_var * 2 * (x - sample_mean) / np.float(N) + dsample_mean / np.float(N)
  dgamma = np.sum(dout * X_norm, axis = 0)
  dbeta = np.sum(dout, axis = 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p # dropout mask. Notice /p!
    out = x * mask # drop!
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(X, W, B, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C_X, H_X, W_X = X.shape
  F, C_W, HH, WW = W.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  
  X_pad = np.zeros((N, C_X, H_X + 2 * pad, W_X + 2 * pad))
  X_pad[:, :, pad:-pad, pad:-pad] = np.copy(X)
  H_out = 1 + np.floor((H_X + 2 * pad - HH) / stride)
  W_out = 1 + np.floor((W_X + 2 * pad - WW) / stride)
  C_out = np.floor(C_X / C_W)
  out = np.zeros((N, F, np.int(H_out), np.int(W_out)))

  for N_iter in range(0, N):
    for F_iter in range(0, F):
      for H_iter in range(0, np.int(H_out)):
        for W_iter in range(0, np.int(W_out)):
          H_start = H_iter * stride
          H_end = H_start + HH
          W_start = W_iter * stride
          W_end = W_start + WW
          out[N_iter, F_iter, H_iter, W_iter] = np.sum(W[F_iter, :, :, :] * X_pad[N_iter, :, H_start : H_end, W_start : W_end]) + B[F_iter]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (X, W, B, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  X, W, B, conv_param = cache
  N, C_X, H_X, W_X = X.shape
  F, C_W, HH, WW = W.shape
  stride = conv_param['stride']
  pad = conv_param['pad']

  X_pad = np.zeros((N, C_X, H_X + 2 * pad, W_X + 2 * pad))
  X_pad[:, :, pad:-pad, pad:-pad] = np.copy(X)
  H_out = 1 + np.floor((H_X + 2 * pad - HH) / stride)
  W_out = 1 + np.floor((W_X + 2 * pad - WW) / stride)

  dx_pad = np.zeros(X_pad.shape)
  dw = np.zeros(W.shape)
  db = np.zeros(B.shape)
  for N_iter in range(0, N):
    for F_iter in range(0, F):
      for H_iter in range(0, np.int(H_out)):
        for W_iter in range(0, np.int(W_out)):
          H_start = H_iter * stride
          H_end = H_start + HH
          W_start = W_iter * stride
          W_end = W_start + WW

          dx_pad[N_iter, :, H_start : H_end, W_start : W_end] += dout[N_iter, F_iter, H_iter, W_iter] * W[F_iter, :, :, :]
          dw[F_iter, :, :, :] += dout[N_iter, F_iter, H_iter, W_iter] * X_pad[N_iter, :, H_start : H_end, W_start : W_end]
          db[F_iter] += dout[N_iter, F_iter, H_iter, W_iter]
  
  dx = dx_pad[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(X, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C_X, H_X, W_X = X.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = 1 + np.floor((H_X - pool_height) / stride)
  W_out = 1 + np.floor((W_X - pool_width) / stride)

  out = np.zeros((N, C_X, H_out, W_out))
  for N_iter in range(0, N):
    for C_iter in range(0, C_X):
      for H_iter in range(0, np.int(H_out)):
        for W_iter in range(0, np.int(W_out)):
          H_start = H_iter * stride
          H_end = H_start + pool_height
          W_start = W_iter * stride
          W_end = W_start + pool_width
          out[N_iter, C_iter, H_iter, W_iter] = np.max(X[N_iter, C_iter, H_start : H_end, W_start : W_end])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (X, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  X, pool_param = cache
  N, C_X, H_X, W_X = X.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = 1 + np.floor((H_X - pool_height) / stride)
  W_out = 1 + np.floor((W_X - pool_width) / stride)

  dx = np.zeros(X.shape)
  for N_iter in range(0, N):
    for C_iter in range(0, C_X):
      for H_iter in range(0, np.int(H_out)):
        for W_iter in range(0, np.int(W_out)):
          H_start = H_iter * stride
          H_end = H_start + pool_height
          W_start = W_iter * stride
          W_end = W_start + pool_width
          index = X[N_iter, C_iter, H_start : H_end, W_start : W_end] == np.max(X[N_iter, C_iter, H_start : H_end, W_start : W_end])
          dx_mask = dx[N_iter, C_iter, H_start : H_end, W_start : W_end]
          dx_mask[index] = dout[N_iter, C_iter, H_iter, W_iter]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape
  num_train = N * H * W
  num_feature = C # channels

  running_mean = bn_param.get('running_mean', np.zeros(num_feature, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(num_feature, dtype=x.dtype))
  x_rshp = x.reshape(num_train, num_feature)
  
  if mode == 'train':
    sample_mean = np.mean(x_rshp, axis = 0)
    sample_var = np.var(x_rshp, axis = 0)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    x_norm = (x_rshp - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_norm + beta
    out = out.reshape(N, C, H, W)

    cache = (x, gamma, beta, sample_mean, sample_var, bn_param)
  elif mode == 'test':
    x_norm = (x_rshp - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_norm + beta
    out = out.reshape(N, C, H, W)

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  x, gamma, beta, sample_mean, sample_var, bn_param = cache
  N, C, H, W = x.shape
  num_train = N * H * W
  num_feature = C # channels
  x = x.reshape(num_train, num_feature)
  dout = dout.reshape(num_train, num_feature)

  eps = bn_param.get('eps', 1e-5)

  MX = np.mean(x, axis = 0)
  VarX = np.var(x, axis = 0)
  VarEps = VarX + eps
  SqVarEps = np.sqrt(VarEps)
  iSqVarEps = SqVarEps ** (-1)
  Xu = x - MX
  Xhat = Xu * iSqVarEps
  out = gamma * Xhat + beta
  
  dy_dXhat = dout * gamma
  dXhat_dXu = dy_dXhat * iSqVarEps

  dXu_dx = dXhat_dXu # 1
  dXu_dMX = -1 * np.sum(dXhat_dXu, axis = 0)
  dMx_dx = dXu_dMX * np.ones((num_train, num_feature)) / np.float(num_train) #2

  dXhat_diSqVarEps = np.sum(dy_dXhat * Xu, axis = 0)
  diSqVarEps_dSqVarEps = -1. / (SqVarEps**(2)) * dXhat_diSqVarEps
  dSqVarEps_dVarEps = 0.5 / np.sqrt(VarEps) * diSqVarEps_dSqVarEps
  dVarEps_dVarX = dSqVarEps_dVarEps
  dVarX_dXu = 1. / np.float(num_train)* 2 * Xu * np.ones((num_train, num_feature)) * dVarEps_dVarX 
  
  dXu_dx_2 = dVarX_dXu # 1
  dXu_dMX_2 = (-1) * np.sum(dVarX_dXu, axis = 0)
  dMx_dx_2 = dXu_dMX_2 * np.ones((num_train, num_feature)) / np.float(num_train) #2
  
  dx = dXu_dx + dXu_dx_2 + dMx_dx + dMx_dx_2
  

  dgamma = np.sum(dout * Xhat, axis = 0)
  dbeta = np.sum(dout, axis = 0)
  dx = dx.reshape(N, C, H, W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
