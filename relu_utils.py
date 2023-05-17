import numpy as np
import numpy.linalg as LA

"""
Some random functions we will need to implement better / improve through the project
"""

"""
class to do F and G multiplicative operations a bit more memory efficiently
"""
class FG_Operators():

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        
        self.P_S = P_S
        self.n = n
        self.d = d
        self.d_diags = d_diags
        self.X = X

    # get matrix F_i
    def F(self, i):
        return self.d_diags[:,i, None] * self.X
    
    # get matrix G_i
    def G(self, i):
        return (2 * self.d_diags[:, i, None] - 1) * self.X
    
    # replace linop F * vec
    def F_multop(self, vec, transpose=False):

        if transpose:
            vec = vec.squeeze()
            assert vec.shape == (self.n,)
            out = np.zeros((2, self.d, self.P_S))
            for i in range(self.P_S):
                out[0,:,i] = self.F(i).T @ vec
                out[1,:,i] -= self.F(i).T @ vec
        else:
            assert vec.shape == (2, self.d, self.P_S)
            out = np.zeros((self.n,))
            for i in range(self.P_S):
                out += self.F(i) @ (vec[0,:,i] - vec[1,:,i])

        return out
    
    # replace linop G * vec
    def G_multop(self, vec, transpose=False):
        
        out = np.zeros((2, self.d if transpose else self.n, self.P_S))

        for i in range(self.P_S):
            for j in range(2):
                out[j,:,i] = (self.G(i).T if transpose else self.G(i)) @ vec[j,:,i]

        return out
    
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