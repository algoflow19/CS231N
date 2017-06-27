from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros((hidden_dim, ))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros((num_classes, ))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2'] 
        reg = self.reg
        (out1, cache1) = affine_relu_forward(X, W1, b1) 
        (scores, cache2) = affine_forward(out1, W2, b2) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        (loss, dout2) = softmax_loss(scores, y)
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5* reg * np.sum(W2 * W2)
        (dx2, dw2, db2) = affine_backward(dout2, cache2) 
        (dx1, dw1, db1) = affine_relu_backward(dx2, cache1)
        grads['W1'] = dw1 + reg * W1
        grads['b1'] = db1
        grads['W2'] = dw2 + reg * W2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        for ii in range(self.num_layers):
            W = 'W' + str(ii + 1)
            b = 'b' + str(ii + 1)
            if ii == 0:
                self.params[W] = weight_scale * np.random.randn(input_dim, hidden_dims[ii])
                self.params[b] = np.zeros((hidden_dims[ii], ))
            elif ii == self.num_layers - 1:
                self.params[W] = weight_scale * np.random.randn(hidden_dims[ii - 1], num_classes)
                self.params[b] = np.zeros((num_classes, ))                    
            else:
                self.params[W] = weight_scale * np.random.randn(hidden_dims[ii - 1], hidden_dims[ii])
                self.params[b] = np.zeros((hidden_dims[ii], ))

        if self.use_batchnorm:
            for ii in range(self.num_layers - 1):
                gamma = 'gamma' + str(ii + 1)
                beta = 'beta' + str(ii + 1)
                self.params[gamma] = np.ones((hidden_dims[ii], ))
                self.params[beta] = np.zeros((hidden_dims[ii], ))
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out = {}
        cache = {}
        drop_cache = {}
        for jj in range(self.num_layers):
            W0 = 'W' + str(jj + 1)
            b0 = 'b' + str(jj + 1)
            W = self.params[W0]
            b = self.params[b0]
            out_save = 'out' + str(jj + 1)
            cache_save = 'cache' + str(jj + 1)
            out_read = 'out' + str(jj)
            cache_read = 'cache' + str(jj)
            if not self.use_batchnorm:
                if jj == 0:
                    (out_tmp, cache_tmp) = affine_relu_forward(X, W, b)
                    if self.use_dropout:
                        (out_tmp, cache_dropout) = dropout_forward(out_tmp, self.dropout_param)
                        drop_cache['dropout_cache' + str(jj + 1)] = cache_dropout
                    out[out_save] = out_tmp
                    cache[cache_save] = cache_tmp
                elif jj == self.num_layers - 1:
                    (scores, cache_tmp) = affine_forward(out[out_read], W, b)
                    cache[cache_save] = cache_tmp
                else:
                    (out_tmp, cache_tmp) = affine_relu_forward(out[out_read], W, b)
                    if self.use_dropout:
                        (out_tmp, cache_dropout) = dropout_forward(out_tmp, self.dropout_param)
                        drop_cache['dropout_cache' + str(jj + 1)] = cache_dropout
                    out[out_save] = out_tmp
                    cache[cache_save] = cache_tmp
            
            if self.use_batchnorm:
                if jj == 0:
                    gamma0 = 'gamma' + str(jj + 1)
                    beta0 = 'beta' + str(jj + 1)
                    gamma = self.params[gamma0]
                    beta = self.params[beta0]
                    (out_tmp, cache_tmp) = affine_bn_relu_forward(X, W, b, gamma, beta, self.bn_params[jj])
                    if self.use_dropout:
                        (out_tmp, cache_dropout) = dropout_forward(out_tmp, self.dropout_param)
                        drop_cache['dropout_cache' + str(jj + 1)] = cache_dropout
                    out[out_save] = out_tmp
                    cache[cache_save] = cache_tmp
                elif jj == self.num_layers - 1:
                    (scores, cache_tmp) = affine_forward(out[out_read], W, b)
                    cache[cache_save] = cache_tmp
                else:
                    gamma0 = 'gamma' + str(jj + 1)
                    beta0 = 'beta' + str(jj + 1)
                    gamma = self.params[gamma0]
                    beta = self.params[beta0]
                    (out_tmp, cache_tmp) = affine_bn_relu_forward(out[out_read], W, b, gamma, beta, self.bn_params[jj])
                    if self.use_dropout:
                        (out_tmp, cache_dropout) = dropout_forward(out_tmp, self.dropout_param)
                        drop_cache['dropout_cache' + str(jj + 1)] = cache_dropout
                    out[out_save] = out_tmp
                    cache[cache_save] = cache_tmp
                
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        dx = {}
        (loss, dout) = softmax_loss(scores, y)
        for kk in reversed(range(self.num_layers)):
            W0 = 'W' + str(kk + 1)
            b0 = 'b' + str(kk + 1)
            W = self.params[W0]
            b = self.params[b0]
            cache0 = 'cache' + str(kk + 1)
            cache1 = cache[cache0]
            dx_save = 'dx' + str(kk)
            dx_read = 'dx' + str(kk + 1)
            loss += 0.5 * self.reg * np.sum(W * W)
            
            if not self.use_batchnorm:
                if kk == self.num_layers - 1:
                    (dx_tmp, dw, db) = affine_backward(dout, cache1)
                else:
                    if self.use_dropout:
                        dx[dx_read] = dropout_backward(dx[dx_read], drop_cache['dropout_cache' + str(kk + 1)])                    
                    (dx_tmp, dw, db) = affine_relu_backward(dx[dx_read], cache1)
                dx[dx_save] = dx_tmp
                grads[W0] = dw + self.reg * W
                grads[b0] = db
                
            if self.use_batchnorm:
                if kk == self.num_layers - 1:
                    (dx_tmp, dw, db) = affine_backward(dout, cache1)
                else:
                    if self.use_dropout:
                        dx[dx_read] = dropout_backward(dx[dx_read], drop_cache['dropout_cache' + str(kk + 1)])                    
                    (dx_tmp, dw, db, dgamma, dbeta) = affine_bn_relu_backward(dx[dx_read], cache1)
                    gamma0 = 'gamma' + str(kk + 1)
                    beta0 = 'beta' + str(kk + 1)
                    grads[gamma0] = dgamma
                    grads[beta0] = dbeta
                dx[dx_save] = dx_tmp
                grads[W0] = dw + self.reg * W
                grads[b0] = db

            

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
