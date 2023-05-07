import numpy as np
import numpy.linalg as LA
from scipy.linalg import block_diag, solve_triangular

from relu_utils import sample_D_matrices

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
    """
    def __init__(self, 
                 m,
                 d,
                 P_S, 
                 rho=1e-5,
                 step=1e-5,
                 beta=1e-5,
                 ):
        self.m = m
        self.d = d
        self.P_S = P_S
        self.rho = rho
        self.step = step
        self.beta = beta

        # final weights of optimized network
        self.u = np.zeros((m,d))
        self.alpha = np.zeros((m,1))

        # flag to ensure predictions only enabled after optimization called
        self.optimized = False

    """
    Optimize cvx neural network by the l-2 squared loss

    :param X - Training data (n x d)
    :param y - Training labels (d x 1)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def optimize(self, X, y, max_iter=100):

        assert len(X.shape) == 2, "X must be 2 dimensional"

        if len(y.shape) == 1:
            y = y[:,None]
        assert len(y.shape) ==2, "Y must be either 1D or 2D"

        n,d = X.shape
        P_S = self.P_S
        r = LA.matrix_rank(X)

        # add bias term to data
        X = np.hstack([X, np.ones((n,1))])
        d += 1

        # sample d matrices
        d_diags = sample_D_matrices(X, P_S)

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

        k = 0 
        while k < max_iter:
            print(f"iter = {k}, loss = (TODO: add)")
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

        ### RECOVER OPTIMAL WEIGHTS

        # recover u1.... u_ms and alpha1 ... alpha_ms
        v_star = v.reshape((P_S*2, d))[:P_S]
        w_star = v.reshape((P_S*2, d))[P_S:P_S*2]
        # critical number of neurons 
        mstar = np.sum(~ np.isclose(LA.norm(v_star, axis=1), 0)) + np.sum(~ np.isclose(LA.norm(w_star, axis=1), 0))

        if self.m > mstar:
            print("m > mstar. Network guranteed not optimal.")

        j = 0
        for i in range(P_S):
            if not np.isclose(LA.norm(v_star[i]), 0):
                self.u[j] = v_star[i] / np.sqrt(LA.norm(v_star[i]))
                self.alpha[j] = np.sqrt(LA.norm(v_star[i]))
                j += 1
            if j == self.m: break
        for i in range(P_S):
            if not np.isclose(LA.norm(w_star[i]), 0):
                self.u[j] = w_star[i] / np.sqrt(LA.norm(w_star[i]))
                self.alpha[j] = - np.sqrt(LA.norm(w_star[i]))
                j += 1
            if j == self.m: break

        print(f"Network has {j} nonzero neurons.")

        self.optimized = True


    """
    Predict classes given new data X

    :param X - Evaluation data (n x d)
    :param max_iter (optional) - max iterations for ADMM algorithm
    """
    def predict(self, X):

        assert self.optimized is True, "Must call .optimize() before applying predictions."
        assert len(X.shape) == 2, "X must be 2 dimensional array"
        assert X.shape[1] == self.d, "X must have same feature size as trained data"
        
        # add bias term to data
        n, d = X.shape
        X = np.hstack([X, np.ones((n,1))])

        # prediction
        y_hat = np.zeros((n,1))
        for j in range(self.m):
            y_hat += np.clip(X @ self.u[j], 0, np.inf) * self.alpha[j]

        return y_hat
    
       


        