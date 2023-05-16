import numpy as np
import numpy.linalg as LA

"""
Some random functions we will need to implement better / improve through the project
"""

"""
Function to sample h_i vectors which create D_i = I(X @ h_i >= 0) matrices given training data X, and number of samples P desired
return: a d x P matrix, where each column i is the random vector h_i
"""
def sample_activation_vectors(X, P, 
                     seed=-1,
                     dist='normal'): #TODO: add typing
     
    assert dist in ['unif', 'normal'], "Sampling must be one of \'unif\', \'normal\'."

    if seed > 0:
        np.random.seed(seed)

    n,d = X.shape

    if dist == 'normal':
        h = np.random.randn(d, P)
    elif dist == 'unif':
        h = np.random.rand(d, P)

    return h


"""
Function to sample diagonals of D_i matrices given training data X, and random vectors h
return: a n x P matrix, where each column i is the diagonal entries for D_i
"""
def get_hyperplane_cuts(X, h): #TODO: add typing

    d_diags = X @ h >= 0
    d_diags.astype("float")

    return d_diags


"""
Calculates the squared loss 1/2||y_hat - y||_2^2
"""
def squared_loss(y_hat, y): #TODO: add typing

    if len(y_hat.shape) == 2:
        y_hat = y_hat.reshape((-1))
    if len(y.shape) == 2:
        y = y.reshape((-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"

    return 0.5 * LA.norm(y_hat - y) ** 2

"""
Calculates accuracy for classification problem
"""
def classifcation_accuracy(y_hat, y):

    if len(y_hat.shape) == 2:
        y_hat = y_hat.reshape((-1))
    if len(y.shape) == 2:
        y = y.reshape((-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = np.round(y_hat)
    return np.sum(y_hat == y) / len(y)

"""
Calculates accuracy for biknary classification problem
"""
def binary_classifcation_accuracy(y_hat, y, binary_class=True):

    if len(y_hat.shape) == 2:
        y_hat = y_hat.reshape((-1))
    if len(y.shape) == 2:
        y = y.reshape((-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = np.sign(y_hat)
    return np.sum(y_hat == y) / len(y)