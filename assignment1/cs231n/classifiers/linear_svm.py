import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #loss = 0.0  
  loss = 0.0
  scores = np.zeros((1,num_classes))
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # lines begin with double "#" are the last version of code!!!!!
  
  ##for i in xrange(num_train):
      #XX = np.tile(X[i,:],(num_classes,1))  # try to use broadcasting
      #scores = np.sum(np.multiply(XX,W.T), axis = 1)
  ##    scores = np.sum(np.multiply(X[i,:],W.T), axis = 1)
    
  ##    if i ==1: print scores
        
      #loss += np.sum(scores - scores[y[i]]) + num_classes -1
      #http://stackoverflow.com/questions/2900084/counting-positive-elements-in-a-list-with-python-list-comprehensions
  ##    scores+=1
  ##    scores[y[i]]-=1  
      #however, this is sum over index, not values, glaube ich  
      #loss+= sum(x < 0 for x in (scores-scores[y[i]]))
  ##    loss+= (scores-scores[y[i]])[scores-scores[y[i]]>0].sum()
    #pass
   ############################################
    # construct a zero loop version
   ############################################
  scores2D = np.zeros((num_train, num_classes)) #used to store dotted scores
  scores1D = np.zeros((num_train,1))            #used to store corrected scores
  #index1D = np.zeros((1,num_classes))
  #index1D = range(num_classes) 
  #scores1D = y[index1D]
 
  scores2D = np.dot(X,W)  
  for i in xrange(num_train):
    scores1D[i,0]=scores2D[i,y[i]]-1            #find the correct scores and fill them into scores1D, the value -1 is because: si-sj+1
    scores2D[i,y[i]]-=1                         # we want at corrected score voxel, the value should be 0, correct score -1 - 
                                                #(correct     score -1) = 0
    
  #scores2D = X.dot(W)
  #http://stackoverflow.com/questions/9497290/how-would-i-sum-a-multi-dimensional-array-in-the-most-succinct-python
  #rewrite summation
  #loss += (scores2D-scores1D)[scores2D-scores1D >0].sum()
  #temp = scores2D-np.tile (scores1D, (1,num_classes))   # for each score minus the corrected score
  temp = scores2D-scores1D                               #broadcasting!!
  #print temp[1,:]
  temp= temp.clip(min=0)  
  #loss += sum(map(sum, (temp)[temp>0]))
  #loss += sum(map(sum, (temp)))
  #loss += (temp)[temp >0].sum()
  loss += sum(sum(x) for x in temp)                   #sum them up
  #loss -= num_train                                  # minus 1 is because in each train, due to the plus 1 above , correct score - correct 
                                                      # score +1 = 1, but it should be 0, therefore, i deduce them at the last minute 
                                                      # ( then I made this also in the for loop to meet intuitive)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #tempBool = np.divide(temp, temp)
  #tempBool = tempBool.clip(max=1,min=0)
  #http://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
  tempBool = np.copy(temp)
  tempBool[tempBool>0] = 1
  for j in xrange(num_train):
    tempBool[j,y[j]] =-1*sum(tempBool[j,:])
    dW += np.reshape (X[j,:],(X.shape[1],1))*tempBool[j,:] 
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dW/= num_train
  dW += reg*W
    
  return loss, dW
