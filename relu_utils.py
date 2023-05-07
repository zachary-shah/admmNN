import numpy as np
import numpy.linalg as LA

"""
Some random functions we will need to implement better / improve through the project
"""

"""
Function to sample D_i matrices given training data X, and number of samples P desired
return: a n x P matrix, where each column i is the diagonal entries for D_i
"""
def sample_D_matrices(X, P): #TODO: add typing
     
    n,d = X.shape

    # sample randomly iid
    h = np.random.randn(d, P)
    d_diags = X @ h >= 0

    # TODO: assert that no duplicate columns

    return d_diags.astype("float")