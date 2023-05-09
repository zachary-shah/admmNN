import numpy as np
import numpy.linalg as LA
from scipy.linalg import block_diag, solve_triangular

from relu_utils import sample_D_matrices, squared_loss, classifcation_accuracy

"""
ReLU solver using ADMM. Implements Algorithm 3.1 of ADMM ReLU Paper.
"""
class Approximate_ReLU_ADMM_Solver():
    """
    Solver attributes

    :param m - Number of hidden layer neurons
    :param d - Feature dimension
    :param P_S - number of samples of D_i matrices #TODO: automatically set this?
    :param rho - fixed penalty parameter (rho > 0)
    :param step - step size constant (step > 0)
    :param beta - augmented lagrangian constant (beta > 0)
    :param bias - True to include bias to weights in first layer
    :param loss_func - function in the form l(y_hat, y) that computes a loss
    :param acc_func - function in the form a(y_hat, y) to compute accuracy of predictions
    :param use_cvxpy - helpful to debug against the CVXPY implimentation. Defaults to False.
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
                 use_cvxpy = False,
                 ):
        self.m = m
        self.P_S = P_S
        self.rho = rho
        self.step = step
        self.beta = beta
        self.bias = bias
        self.loss_func = loss_func
        self.acc_func = acc_func
        self.d = None

        # optimal weights of C-ReLU
        self.v = None
        self.w = None
        # for forming D matrices as diag([X h >= 0])
        self.h = None

        # optimal weights of NC-ReLU
        self.u = None
        self.alpha = None

        # flag to ensure predictions only enabled after optimization called
        self.optimized = False

    """
    Optimize cvx neural network by the l-2 squared loss

    :param X - Training data (n x d)
    :param y - Training labels (d x 1)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def optimize(self, X, y, max_iter=100, verbose=False):

        # TODO: Normalize X? 

        assert len(X.shape) == 2, "X must be 2 dimensional"
        if len(y.shape) == 1:
            y = y[:,None]
        assert len(y.shape) == 2, "Y must be either 1D or 2D"

        n, d = X.shape
        P_S = self.P_S
        r = LA.matrix_rank(X)

        # add bias term to data
        if self.bias:
            X = np.hstack([X, np.ones((n,1))])
            d += 1

        # prepare final weights of optimized network
        self.d = d
        self.u = np.zeros((self.m,self.d))
        self.alpha = np.zeros((self.m,1))

        # sample d matrices
        # TODO: sample diags with np.random.choice if X is standard normal 
        d_diags, h = sample_D_matrices(X, P_S)
        self.h = h

        if use_cvxpy:
            import cvxpy as cp
            
            # Variables
            v = cp.Variable((d * P_S))
            w = cp.Variable((d * P_S))
            
            # Construct all possible data enumerations (n x P_S * d)
            F = np.hstack([d_diags[:, i, None] * X for i in range(P_S)])

            # Objective Function
            def obj_func(F, y, v, w):
                l2_term = cp.sum_squares(
                    (F @ (v - w)) - y[:, 0]
                )
                group_sparsity = 0
                for i in range(P_S):
                    group_sparsity += cp.norm2(v[i*d:(i+1)*d])
                    group_sparsity += cp.norm2(w[i*d:(i+1)*d])

                return l2_term + self.beta * group_sparsity
            
            # Solve via cvxpy
            prob = cp.Problem(cp.Minimize(obj_func(F, y, v, w)),
                              [((2 * d_diags[:, i, None] - 1) * X) @ v[i*d:(i+1)*d] >= 0 for i in range(P_S)] + \
                              [((2 * d_diags[:, i, None] - 1) * X) @ w[i*d:(i+1)*d] >= 0 for i in range(P_S)])
            prob.solve()
            
            # Grab optimal values 
            self.v = np.reshape(v.value, (P_S, d), order='C')
            self.w = np.reshape(w.value, (P_S, d), order='C')
            train_loss = [0] * max_iter
            train_acc = [0] * max_iter

        else:

            # F here is n x (2d*P_S)
            F = np.hstack([np.hstack([np.diag(d_diags[:,i]) @ X for i in range(P_S)]),
                np.hstack([-np.diag(d_diags[:,i]) @ X for i in range(P_S)[::-1]])])

            # G is block diagonal 2*n*P_S x 2*d*P_s
            Glist = [(2 * np.diag(d_diags[:,i]) - np.eye(n)) @ X for i in range(P_S)]
            G = block_diag(*Glist * 2)

            ### INITIALIZATIONS OF OPTIMIZATION VARIABLES 

            # u contains u1 ... uP, z1... zP in one long vector
            u = np.zeros((2 * d * P_S, 1))
            # v contrains v1 ... vP, w1 ... wP in one long vector
            v = np.zeros((2 * d * P_S, 1))

            # slacks s1 ... sP, t1 ... tP
            s = np.zeros((2 * n * P_S, 1))
            for i in range(P_S):
                # s_i = G_i v_i
                s[i*n:(i+1)*n] = Glist[i] @ v[i*d:(i+1)*d]
                s[(i+P_S)*n:(i+P_S+1)*n] = Glist[i] @ v[(i+P_S)*d:(i+P_S+1)*d]

            # dual variables
            # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
            lam = np.zeros((2 * d * P_S, 1))
            # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
            nu = np.zeros((2 * n * P_S, 1))

            ### PRECOMPUTATIONS

            # for u update: A in 2dP_s x 2dP_s
            A = np.eye(2*d*P_S) + F.T @ F / self.rho + G.T @ G
            # cholesky factorization
            L = LA.cholesky(A)
            # extra precompute for u update step
            b_1 = F.T @ y / self.rho

            ## ITERATIVE UPDATES 

            # keep track of losses
            train_loss, train_acc = [], []

            k = 0 
            while k < max_iter:

                # keep track of losses
                y_hat = F @ u
                train_loss.append(self.loss_func(y_hat, y))
                train_acc.append(self.acc_func(y_hat, y))
                if verbose: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

                # first, conduct the primal update on u (u1...uP, z1...zP)
                b = b_1 + v - lam + G.T @ (s - nu)
                bhat = solve_triangular(L, b, lower=True)
                u = solve_triangular(L.T, bhat, lower=False)

                # second, perform updates of v and s (TODO: parallelize v and s updates)
                # upates on v = (v1...vP, w1...wP)
                for i in range(2 * P_S):
                    inds = np.arange(d*i, d*(i+1))
                    v[inds] = np.maximum(1 - self.beta/(self.rho * LA.norm(u[inds] + lam[inds])), 0) * (u[inds] + lam[inds])
                # updates on s = (s1...sP, t1...tP)
                for i in range(2 * P_S):
                    s[i*n:(i+1)*n] = np.maximum(Glist[i % P_S] @ u[i*d:(i+1)*d] + nu[i*n:(i+1)*n], 0)

                # finally, perform dual updates
                lam += self.step / self.rho * (u - v)
                nu += self.step / self.rho * (G @ u - s)

                # iter step 
                k += 1        

            # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
            self.v = v.reshape((P_S*2, d))[:P_S]
            self.w = v.reshape((P_S*2, d))[P_S:P_S*2]

        # recover u1.... u_ms and alpha1 ... alpha_ms as Optimal Weights of NC-ReLU Problem
        self._optimal_weights_transform()

        self.optimized = True
        self.train_loss = np.array(train_loss)
        self.train_acc = np.array(train_acc)

    """
    Predict classes given new data X

    :param X - Evaluation data (n x d)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def predict(self, X, weights="C-ReLU"):

        assert self.optimized is True, "Must call .optimize() before applying predictions."
        assert len(X.shape) == 2, "X must be 2 dimensional array"
        assert X.shape[1] == self.d - int(self.bias), f"X must have same feature size as trained data (d={self.d - int(self.bias)})"
        assert weights in ["NC-ReLU", "C-ReLU"], f"Weights options are either \"NC-ReLU\" for weights of non-convex problem, or \"C-ReLU\" for weights of convex problem"
        
        # add bias term to data
        n, d = X.shape
        if self.bias:
            X = np.hstack([X, np.ones((n,1))])

        y_hat = np.zeros((n,1))
        
        # prediction using weights for equivalent nonconvex problem
        if weights == "NC-ReLU": 
            for j in range(self.m):
                y_hat += np.clip(X @ self.u[j][:,None], 0, np.inf) * self.alpha[j]
        # prediction using weights for solved convex problem
        elif weights == "C-ReLU": 
            for i in range(self.P_S):
                D_i = np.diag(X @ self.h[:,i] >= 0).astype('float')
                y_hat += D_i @ X @ (self.v[i][:,None] - self.w[i][:,None])
        else:
            raise NotImplementedError

        return y_hat
    
    """
    Given optimal v^*, w^* of convex problem (Eq (2.1)), derive the optimal weights u^*, alpha^* of the non-convex probllem (Eq (2.1))
    Applies Theorem 1 of Pilanci, Ergen 2020
    TODO: fix function. I don't think its behaving the way it should
    - what is 1i and 2i indices of Theorem 1?
    """
    def _optimal_weights_transform(self):

        assert self.v is not None
        assert self.w is not None

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

        print(f"Network of width {self.m} has {j} nonzero neurons for non-convex weights.")


    
       


        