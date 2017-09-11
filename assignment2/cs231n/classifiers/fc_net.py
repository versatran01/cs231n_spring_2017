import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
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

        #######################################################################
        # TODO: Initialize the weights and biases of the two-layer net.
        # Weights should be initialized from a Gaussian with standard
        # deviation equal to weight_scale, and biases should be initialized to
        # zero. All weights and biases should be stored in the dictionary
        # self.params, with first layer weights and biases using the keys
        # 'W1' and 'b1' and second layer weights and biases using the keys
        # 'W2' and 'b2'.
        #######################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim,
                                                           hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim,
                                                           num_classes)
        self.params['b2'] = np.zeros(num_classes)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

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
        #######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing   #
        # the class scores for X and storing them in the scores variable.     #
        #######################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        a1, a1_cache = affine_relu_forward(X, W1, b1)
        a2, a2_cache = affine_forward(a1, W2, b2)

        scores = a2

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the
        # loss in the loss variable and gradients in the grads dictionary.
        # Compute data loss using softmax, and make sure that grads[k] holds
        # the gradients for self.params[k]. Don't forget to add L2
        # regularization!
        # NOTE: To ensure that your implementation matches ours and you pass
        # the automated tests, make sure that your L2 regularization includes
        # a factor of 0.5 to simplify the expression for the gradient.
        #######################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2

        dx1, dw1, db1 = affine_relu_backward(dx2, a1_cache)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

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

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0
          then the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch
          normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed
          using this datatype. float32 is faster but less accurate, so you
          should use float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deterministic so we can
          gradient check the model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ########################################################################
        # TODO: Initialize the parameters of the network, storing all values in
        # the self.params dictionary. Store weights and biases for the first
        # layer in W1 and b1; for the second layer use W2 and b2,
        # etc. Weights should be initialized from a normal distribution with
        # standard deviation equal to weight_scale and biases should be
        # initialized to zero.
        #
        # When using batch normalization, store scale and shift parameters
        # for the first layer in gamma1 and beta1; for the second layer use
        # gamma2 and beta2, etc. Scale parameters should be initialized to
        # one and shift parameters should be initialized to zero.
        #######################################################################
        dim_in = input_dim
        for i, dim_out in enumerate(hidden_dims):
            n = i + 1
            self.set_W(n, weight_scale * np.random.randn(dim_in, dim_out))
            self.set_b(n, np.zeros(dim_out))
            if self.use_batchnorm:
                self.set_gamma(n, np.ones(dim_out))
                self.set_beta(n, np.zeros(dim_out))
            dim_in = dim_out

        dim_out = num_classes
        self.set_W(self.num_layers, weight_scale * np.random.randn(dim_in,
                                                                   dim_out))
        self.set_b(self.num_layers, np.zeros(dim_out))
        #######################################################################
        #                        END OF YOUR CODE                             #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the
        # mode (train / test). You can pass the same dropout_param to each
        # dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the
        # forward pass of the first batch normalization layer,
        # self.bn_params[1] to the forward pass of the second batch
        # normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in
                              range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def W(self, i):
        return self.params[_make_key('W', i)]

    def b(self, i):
        return self.params[_make_key('b', i)]

    def set_W(self, i, W):
        self.params[_make_key('W', i)] = W

    def set_b(self, i, b):
        self.params[_make_key('b', i)] = b

    def gamma(self, i):
        return self.params[_make_key('gamma', i)]

    def beta(self, i):
        return self.params[_make_key('beta', i)]

    def set_gamma(self, i, gamma):
        self.params[_make_key('gamma', i)] = gamma

    def set_beta(self, i, beta):
        self.params[_make_key('beta', i)] = beta

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net,
        # computing the class scores for X and storing them in the scores
        # variable.
        #
        # When using dropout, you'll need to pass self.dropout_param to each
        # dropout forward pass.
        #
        # When using batch normalization, you'll need to pass self.bn_params[0]
        # to the forward pass for the first batch normalization layer, pass
        # self.bn_params[1] to the forward pass for the second batch
        # normalization layer, etc.
        #######################################################################
        x = X
        ar_cache = {}

        n = self.num_layers
        for i in range(1, n):
            Wi = self.W(i)
            bi = self.b(i)

            if self.use_batchnorm:
                gammai = self.gamma(i)
                betai = self.beta(i)
                z, cache = affine_bn_relu_forward(x, Wi, bi, gammai, betai,
                                                  self.bn_params[i - 1])
            else:
                z, cache = affine_relu_forward(x, Wi, bi)

            # update
            x = z
            ar_cache[i] = cache

        # last affine layer
        Wn = self.W(n)
        bn = self.b(n)
        scores, cache = affine_forward(x, Wn, bn)
        ar_cache[n] = cache

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store
        # the loss in the loss variable and gradients in the grads
        # dictionary. Compute data loss using softmax, and make sure that
        # grads[k] holds the gradients for self.params[k]. Don't forget to add
        # L2 regularization!
        #
        # When using batch normalization, you don't need to regularize the
        # scale and shift parameters.
        #
        # NOTE: To ensure that your implementation matches ours and you pass
        # the automated tests, make sure that your L2 regularization includes
        # a factor of 0.5 to simplify the expression for the gradient.
        #######################################################################
        loss, dscores = softmax_loss(scores, y)

        # retrieve params
        n = self.num_layers
        Wn = self.W(n)
        loss += 0.5 * self.reg * np.sum(Wn * Wn)  # add reg

        # backprop  last affine layer
        dx, dw, db = affine_backward(dscores, ar_cache[n])
        grads[_make_key('W', n)] = dw + self.reg * Wn
        grads[_make_key('b', n)] = db

        dz = dx

        for i in range(n - 1, 0, -1):
            # retrieve params
            Wi = self.W(i)

            # add reg to loss
            loss += 0.5 * self.reg * np.sum(Wi * Wi)

            if self.use_batchnorm:
                # backprop affine bn relu layer
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dz,
                                                                    ar_cache[i])
                grads[_make_key('gamma', i)] = dgamma
                grads[_make_key('beta', i)] = dbeta
            else:
                # backprop affine relu layer
                dx, dw, db = affine_relu_backward(dz, ar_cache[i])

            # update grads
            dw += self.reg * Wi
            grads[_make_key('W', i)] = dw
            grads[_make_key('b', i)] = db

            # update dout
            dz = dx

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        return loss, grads


def _make_key(p, i):
    return p + str(i)


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    af_out, af_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(af_out, gamma, beta, bn_param)
    rl_out, rl_cache = relu_forward(bn_out)
    cache = (af_cache, bn_cache, rl_cache)
    return rl_out, cache


def affine_bn_relu_backward(dout, cache):
    af_cache, bn_cache, rl_cache = cache
    dbn = relu_backward(dout, rl_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(da, af_cache)
    return dx, dw, db, dgamma, dbeta
