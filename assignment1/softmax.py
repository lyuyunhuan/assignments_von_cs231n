import numpy as np
from random import shuffle
from math import exp, log

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
  ##################################
  # softMax is a modification abased on the traditional SVM model, except adding an exponantial, normalize, and negative log
  ##################################
  for i in xrange(num_train):
    scores = np.zeros((num_classes,1))
    scores = X[i,:].dot(W)
    #print scores.shape[0]
    expScores = np.zeros((1,num_classes))
    for qq in xrange (num_classes):
        expScores[0,qq] = exp(scores[qq])       
    sumScores = np.sum(expScores)
    #print sumScores
    derLog = ( sumScores/(expScores[0,y[i]]) )   #this is for the derivative of log: ( 1 over x)
    for j in xrange(num_classes):
        if j == y[i]:                            #for the class of corrected lable, the whole derivative should go through another part:
                                                 # X over (X+Y+Z) = deri(X) / (X+Y+Z) + constant * deri(X) / (X+Y+Z)^2
                                                 # and accidentally, deri(X) = constant = X in the class of corrected lable~~
             loss += -log (expScores[0,y[i]]/sumScores)
             dW[:,j] += (X[i,:].T) * -derLog * ((expScores[0,y[i]])/sumScores +
                                         (-1) * (expScores[0,y[i]]) *(expScores[0,y[i]]) /(sumScores*sumScores) ) 
            
        else:                                     # this is the most generalized part, every classes should go through this:
                                                  # it is the derivatie of constant:expScores[0,y[i]], over (X+Y+Z) = 
                                                  # constant * deri(X) / (X+Y+Z)^2
             dW[:,j] += (X[i,:].T)  * -derLog *  (-1) * (expScores[0,y[i]]) * (expScores[0,j])/  (sumScores*sumScores) 
                                                  
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  loss /= num_train
  dW/= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

