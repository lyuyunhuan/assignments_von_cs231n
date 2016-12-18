import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    H, C = W2.shape
    
    h1 = np.maximum ( 0, X.dot(W1) + b1.T)             #simply using np.maximum to eliminate negative values in hidden layer 1
    scores = h1.dot(W2) + b2.T                         # note that: scores are not the values pass softMax, they are data vor the softMax
    #print h1
    #print scores
    # Compute the forward pass
    #scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    
    score1 = np.exp(scores)                           # these steps are softMax
    score2 = np.sum(score1, axis = 1)
    score2 = np.reshape(score2, (N,1))
    score3 = score1 / score2
    
    loss = 0
    for i in xrange(N):                               # add loss values from each sample, don't forget to divide with num_train
        loss += -np.log(score3[i,y[i]])/N
 
    #loss += sum (sum(W2.dot(W2.T))) * reg *0.5
    loss += sum( sum(W2**2) ) * reg *0.5              # regularization 2, note that: L2 required for each single WEIGHTS
    loss += sum( sum(W1**2) ) * reg *0.5
    #print loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dh1 = np.zeros_like(h1)
    
    #print db1
    
    for i in xrange(N):                                  # this for loop is for the second layer derivative calculation: dW2 and db2
         for j in xrange(C):                             # the whole process could reference the one-layer softMax net that I generated
                 if j == y[i]:                           # earlier.
                        dW2[:,j] -= (h1[i,:].T)          # the biggest difference is that the dscore3 matrix affect both dW2 and db2 in the 
                        db2[j] -= 1                      # same way!!!
                 dW2[:,j] += (h1[i,:].T) * score3[i,j]
                 db2[j] += 1 * score3[i,j]
    
    #db2 = np.sum(scores,axis=0)
    
    h1Bool = np.copy(h1)                                 # h1Bool is used to backward compansate the Relu activation function, 
    h1Bool[h1Bool > 0] = 1                               # all positive are set to one, and zeros remain zero
    
    dscore3 = score3                                     # dscore3 is to visualize the dscore which I already use for convinience 
    for i in xrange(N):                                  # dscore3 is the backward compansate the softMax activation function
        dscore3[i,y[i]]-=1
    
    for i in xrange(N):                                  # dh1 in layer one is wie the dscore3 in layer two, 
        dh1[i,:] = dscore3[i,:].dot(W2.T) * h1Bool[i,:]  # to calcuate dh1, we need to go through backward through, dscore3, W2 and 
    #print h1Bool                                        # h1Bool
    
    for i in xrange(N):                                  # after we have generate the dh1, the calculation of the dW1 in the two layer net
         dW1 += np.tile(X[i,:], (H,1)).T * dh1[i,:]      # could be simplified as a single layer net
    
    db1 = np.sum(dh1, axis=0)                            # due to the X for db1 is always one, db1 is the sum of the dh1 along the axis=0
    #print db1
    
    dW1/= N                                              # don't forget divided by the train_num
    dW2/= N
    db1/= N
    db2/= N
    
    dW1 += reg*W1                                        # and don't forget the regularization part
    dW2 += reg*W2
    
    grads = {'W1':dW1, 'W2':dW2, 'b1':db1, 'b2':db2}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []         
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      
      indexArr = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[[indexArr]][:]
      y_batch = y[[indexArr]]
    
      #pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)                    # throw batches to the loss function to obtain the loss 
      loss_history.append(loss)                                               # and the gradients of Ws, and bs

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      
      self.params['W1'] -= 1e-0* grads['W1'] * loss * learning_rate           # use the obtained gradients to up date Ws, and bs
      self.params['W2'] -= 1e-0* grads['W2'] * loss * learning_rate           # at first I thought that the updating rate should be
      self.params['b1'] -= 1e-0* grads['b1'] * loss * learning_rate           # multiplied by std(see _init_ part), but I was wrong
      self.params['b2'] -= 1e-0* grads['b2'] * loss * learning_rate           # I also find out that when the update rate is too small
      #pass                                                                   # the loss scores are just ocsilate, not decay
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:                                           # it stands for which iteration is right now
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)                                   # put accuracy data into history
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay                                  # decay learning rate after each epoch

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None                                                             # just initializing
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    H, C = W2.shape
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    hl = np.maximum ( 0, X.dot(W1) + b1.T) 
    scores = hl.dot(W2) + b2.T
    score1 = np.exp(scores)                                                   # these steps are softMax
    score2 = np.sum(score1, axis = 1)
    score2 = np.reshape(score2, (N,1))
    score3 = score1 / score2
    
    y_pred = np.argmax(score3, axis=1)                                        # return the label of correct class for each Sample
    
    #pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


