import numpy as np
import numpy.linalg as LA

"""
Some random functions we will need to implement better / improve through the project
"""

"""
Function to sample D_i matrices given training data X, and number of samples P desired
return: a n x P matrix, where each column i is the diagonal entries for D_i
"""
def sample_D_matrices(X, P, seed=-1): #TODO: add typing
     
    if seed > 0:
        np.random.seed(seed)

    n,d = X.shape

    # sample randomly iid
    h = np.random.randn(d, P)
    d_diags = X @ h >= 0

    # TODO: assert that no duplicate columns

    return d_diags.astype("float"), h


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