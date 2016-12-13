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
  scores2D = np.exp (np.dot(X,W))

  for i in xrange(num_train):
    scores = np.zeros((num_classes,1))
    scores = X[i,:].dot(W)
    #print scores.shape[0]
    expScores = np.zeros((1,num_classes))
    for qq in xrange (num_classes):
        expScores[0,qq] = exp(scores[qq])       
    #if expScores[0,y[i]] == scores2D[i,y[i]]: print i
    sumScores = np.sum(expScores)
    #if i ==2: print expScores[0,y[2]]
    #print sumScores
    derLog = ( sumScores/(expScores[0,y[i]]) )   #this is for the derivative of log: ( 1 over x)
    for j in xrange(num_classes):
        if j == y[i]:                            #for the class of corrected lable, the whole derivative should go through another part:
                                                 # X over (X+Y+Z) = deri(X) / (X+Y+Z) + constant * deri(X) / (X+Y+Z)^2
                                                 # and accidentally, deri(X) = constant = X in the class of corrected lable~~
             loss += -log (expScores[0,y[i]]/sumScores)
             #dW[:,j] += (X[i,:].T) * -derLog * ((expScores[0,y[i]])/sumScores +
             #                            (-1) * (expScores[0,y[i]]) *(expScores[0,y[i]]) /(sumScores*sumScores) ) 
             dW[:,j] -= (X[i,:].T)
             #dW[:,j] += (X[i,:].T) / sumScores *(expScores[0,y[i]])## / sumScores
            
        #else:                                     # this is the most generalized part, every classes should go through this:
                                                  # it is the derivatie of constant:expScores[0,y[i]], over (X+Y+Z) = 
                                                  # constant * deri(X) / (X+Y+Z)^2
             #dW[:,j] += (X[i,:].T)  * -derLog *  (-1) * (expScores[0,y[i]]) * (expScores[0,j])/  (sumScores*sumScores) 
        dW[:,j] += (X[i,:].T) /  (sumScores) * (expScores[0,j])##/  (sumScores) 
        #dW[:,j] += (X[i,:].T) /  (sumScores) * (scores2D[i,j])
                                                  
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores2D = np.zeros((num_train, num_classes)) #used to store dotted scores
  scores1D = np.zeros((num_train,1))            #used to store corrected scores
  scores2D = np.exp (np.dot(X,W))
  alle2D = np.ones((num_train,num_classes))
  labeled2D = np.zeros((num_train, num_classes))
  rowSumScore = np.zeros((num_train))
  rowSumScore = np.sum(scores2D, axis = 1)  
  
  #print scores2D[2,:]                          #print *4 are only for check
  #print rowSumScore[2]
    
  indexInsert = np.arange(num_train)
  scores1D[indexInsert,0] = scores2D[indexInsert,y[indexInsert]]# / (rowSumScore[indexInsert,0])  #using array indexing
  scores1D[indexInsert,0] /= rowSumScore[indexInsert]  # I still dont know why sometimes indexing should only be 1D
  #print y[2]
  #print scores1D[2,0]  
  #rowSumScore[indexInsert,0] = log( scores1D[indexInsert,0])
  loss += sum(sum(-np.log(scores1D)))            # don't forget / num_train, and regulerization
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #print rowSumScore.shape
  #print scores2D[2,y[2]]
  for j in xrange(num_train):
    labeled2D[j,y[j]] = 1
    #tempBool[j,y[j]] =-1*sum(tempBool[j,:])  # calculate how many final scores, max(~~,0) are more than 0, add the number to the correct
                                             # label element, because it is the times that the corrected scores be used
    #dW += np.reshape (X[j,:],(X.shape[1],1))*np.log(scores2D[j,:])*(np.exp(scores2D[j,y[j]]))*(np.exp(scores2D[j,:]))/(rowSumScore[j]) /(scores2D[j,y[j]])#/  np.reshape(rowSumScore[j,0], (X.shape[1],1)) #*rowSumScore[j,0])  # broadcasting, out-product  
    #tempVa = scores2D[j,y[j]]
    dW -= np.reshape (X[j,:],(X.shape[1],1)) * labeled2D[j,:]
    dW += np.reshape (X[j,:],(X.shape[1],1)) * alle2D[j,:]  * scores2D[j,:] / (rowSumScore[j])## / (rowSumScore[j])
    
  #dW += X[indexInsert,:].T*scores2D[indexInsert,:]#*(-np.log(scores1D[indexInsert,0])) * (expScores[0,y[indexInsert]])*  (expScores[0,indexInsert])/  (rowSumScore[indexInsert,0]*rowSumScore[indexInsert,0])
   
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  dW/= num_train
  dW += reg*W

  return loss, dW