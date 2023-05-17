import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler
from time import perf_counter

from relu_utils import sample_activation_vectors, get_hyperplane_cuts, squared_loss, classifcation_accuracy
from optimizers import IMPLEMENTED_OPTIMIZERS, admm_optimizer

"""
Shell structure for 2 Layer Convex ReLU Solver
"""
class Approximate_2_Layer_ReLU():
    """
    Solver attributes
    :param m - Number of hidden layer neurons for equivalent NC-ReLU Problem (not used if using C-ReLU weights)
    :param d - Feature dimension of data
    :param P_S - number of samples of ReLU activation patterns (D_i matrices) #TODO: automatically set this?
    :param rho - fixed penalty parameter (rho > 0)
    :param step - step size constant (step > 0)
    :param beta - augmented lagrangian constant (beta > 0)
    :param bias - True to include bias to weights in first layer
    :param loss_func - function in the form l(y_hat, y) that computes a loss
    :param acc_func - function in the form a(y_hat, y) to compute accuracy of predictions
    :param optimizer - choice of optimizer, must be element of optimizers.IMPLEMENTED_OPTIMIZERS 
    :param standardize_data - (Optional) True to standardize features using sclearn.preprocessing.StandardScaler 
    :param seed - (Optional) random seed
    """
    def __init__(self, 
                 m,
                 P_S, 
                 rho=1e-5,
                 step=1e-5,
                 beta=1e-5,
                 bias=True,
                 loss_func = squared_loss,
                 acc_func = classifcation_accuracy,
                 optimizer = admm_optimizer,
                 standardize_data = False,
                 seed=-1,
                 ):
        
        assert optimizer in IMPLEMENTED_OPTIMIZERS, f"Optimizer not implemented, must be one of {IMPLEMENTED_OPTIMIZERS}"

        # problem dimensions
        self.m = m
        self.P_S = P_S
        self.d = None

        # hyperparameters
        self.rho = rho
        self.step = step
        self.beta = beta

        # other options
        self.bias = bias
        self.optimizer = optimizer
        self.standardize_data = standardize_data
        self.seed = seed
        self.loss_func = loss_func
        self.acc_func = acc_func
        self.optimizer = optimizer

        # random vectors used for forming D matrices as diag([X h >= 0])
        self.h = None

        # optimal weights of C-ReLU
        self.v = None
        self.w = None
        
        # optimal weights of NC-ReLU
        self.u = None
        self.alpha = None

        # optimization metrics for keeping track of performance
        self.metrics = {}

        # flag to ensure predictions only enabled after optimization called
        self.optimized = False

    """
    Optimize cvx neural network by the l-2 squared loss

    :param X - Training data (n x d)
    :param y - Training labels (d x 1)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def optimize(self, X, y, 
                 max_iter=100, 
                 verbose=False):

        assert len(X.shape) == 2, "X must be 2 dimensional"
        if len(y.shape) == 1:
            y = y[:,None]
        assert len(y.shape) == 2, "Y must be either 1D or 2D"

        n, d = X.shape
        self.d = d
        P_S = self.P_S
        r = LA.matrix_rank(X)

        # cleanly preprocess data
        X = self._preprocess_data(X)

        # sample random vectors and get diagonal entries of D_i matrices
        self.h = sample_activation_vectors(X, P_S, seed=self.seed, dist='normal')

        t_start = perf_counter()

        self = self.optimizer(self, X, y, max_iter, verbose=verbose)
        
        self.metrics["solve_time"] = perf_counter() - t_start

        # recover u1.... u_ms and alpha1 ... alpha_ms as Optimal Weights of NC-ReLU Problem
        self._optimal_weights_transform(verbose=verbose)

        self.optimized = True
        

    """
    Predict classes given new data X

    :param X - Evaluation data (n x d)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def predict(self, X, weights="C-ReLU"):

        assert self.optimized is True, "Must call .optimize() before applying predictions."
        assert len(X.shape) == 2, "X must be 2 dimensional array"
        assert X.shape[1] == self.d, f"X must have same feature size as trained data (d={self.d})"
        assert weights in ["NC-ReLU", "C-ReLU"], f"Weights options are either \"NC-ReLU\" for weights of non-convex problem, or \"C-ReLU\" for weights of convex problem"

        X = self._preprocess_data(X)

        y_hat = np.zeros((X.shape[0],1))
        
        # prediction using weights for equivalent nonconvex problem
        if weights == "NC-ReLU": 
            print("WARNING: NC-RelU weights not correctly implmeneted yet.")
            for j in range(self.m):
                y_hat += np.clip(X @ self.u[j][:,None], 0, np.inf) * self.alpha[j]

        # prediction using weights for solved convex problem
        elif weights == "C-ReLU": 
            d_diags = get_hyperplane_cuts(X, self.h)
            for i in range(self.P_S):
                y_hat += (d_diags[:,i, None] * X) @ (self.v[i][:,None] - self.w[i][:,None])
        else:
            raise NotImplementedError

        return y_hat
    
    """
    Get metrics from solver
    """
    def get_metrics(self):
        assert self.optimized is True, "Must call .optimize() to solve problem first."
        return self.metrics
    
    """
    Preprocess the data X based on desired inputs (standardization, bias, etc)
    """
    def _preprocess_data(self, X):

        # standardize data if desired
        if self.standardize_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # add bias term to data
        n, d = X.shape
        if self.bias:
            X = np.hstack([X, np.ones((n,1))])

        return X

    """
    Given optimal v^*, w^* of convex problem (Eq (2.1)), derive the optimal weights u^*, alpha^* of the non-convex probllem (Eq (2.1))
    Applies Theorem 1 of Pilanci, Ergen 2020
    TODO: fix function. I don't think its behaving the way it should
    - what is 1i and 2i indices of Theorem 1?
    """
    def _optimal_weights_transform(self, verbose=False):

        assert self.v is not None
        assert self.w is not None

        self.u = np.zeros((self.m,self.d + int(self.bias)))
        self.alpha = np.zeros((self.m,1))

        # critical number of neurons 
        mstar = np.sum(~ np.isclose(LA.norm(self.v, axis=1), 0)) + np.sum(~ np.isclose(LA.norm(self.w, axis=1), 0))
        if self.m > mstar:
            print("m > mstar. Network guranteed not optimal.")

        i,j = 0,0
        while i < self.P_S and j < self.m:
            if not np.isclose(LA.norm(self.v[i]), 0):
                self.u[j] = self.v[i] / np.sqrt(LA.norm(self.v[i]))
                self.alpha[j] = np.sqrt(LA.norm(self.v[i]))
                j += 1
            i += 1
        i = 0
        while i < self.P_S and j < self.m:
            if not np.isclose(LA.norm(self.w[i]), 0):
                self.u[j] = self.w[i] / np.sqrt(LA.norm(self.w[i]))
                self.alpha[j] = - np.sqrt(LA.norm(self.w[i]))
                j += 1
            i += 1

        if verbose: print(f"Network of width {self.m} has {j} nonzero neurons for non-convex weights.")
    
       


        