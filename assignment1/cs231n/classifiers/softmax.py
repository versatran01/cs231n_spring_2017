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

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful  #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                                        #
    ##########################################################################
    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
        # Compute score of single input
        f_i = np.dot(X[i], W)  # 1xD * DxC = 1xC
        # Normalize to avoid numerical instability
        f_i -= np.amax(f_i)

        E = np.exp(f_i)
        sum_E = np.sum(E)
        loss += -np.log(E[y[i]] / sum_E)

        for j in range(C):
            dW[:, j] += (E[j] / sum_E - (j == y[i])) * X[i]

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ###########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.#
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                                         #
    ###########################################################################

    N = X.shape[0]
    C = W.shape[1]

    F = np.dot(X, W)  # NxD * DxC = NxC
    F -= np.amax(F, axis=1, keepdims=True)
    E = np.exp(F)  # NxC
    S = np.sum(E, axis=1, keepdims=True)  # 1xN
    P = np.divide(E, S)  # NxC

    loss = np.sum(-np.log(P[np.arange(N), y]))

    ind = np.zeros_like(P)  # NxC
    ind[np.arange(N), y] = 1
    dW = np.dot(X.T, P - ind)  # DxN * NxC = DxC

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    dW /= N
    dW += reg * W

    return loss, dW
