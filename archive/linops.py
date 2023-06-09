import sigpy as sp
import numpy as np
import numpy.linalg as LA
import time
from scipy.linalg import solve_triangular
from linops import F_linop, G_linop

from relu_utils import get_hyperplane_cuts


# The sigpy library uses this linear operator abstraction, which is aweosme.
# Basically, when applying some linear function A to an input x, you can of course 
# use the matrix representation of A, but it is often not as efficient as exploting 
# the matrix A's structure. In our case, G, F are repetative, and have a structure 
# that is worth exploting.
class G_linop(sp.linop.Linop):

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        super().__init__((2, n, P_S), (2, d, P_S))

        # Build linop
        M = sp.linop.MatMul(self.ishape, X)
        W = sp.linop.Multiply(M.oshape, (2.0 * d_diags[None, ...] - 1))
        self.linop = W * M

    def _apply(self, input):
        return self.linop * input
    
    def _adjoint_linop(self):
        return self.linop.H
    
    def _normal_linop(self):
        return self.linop.H * self.linop
    
class F_linop(sp.linop.Linop):

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        super().__init__((n,), (2, d, P_S))

        # Build linop
        diag_tot = np.concatenate((1.0 * d_diags[None, ...], -1.0 * d_diags[None, ...]), axis=0)
        M = sp.linop.MatMul(self.ishape, X)
        W = sp.linop.Multiply(M.oshape, diag_tot)
        S = sp.linop.Sum(W.oshape, axes=(0, 2))

        self.linop = S * W * M

    def _apply(self, input):
        return self.linop * input
    
    def _adjoint_linop(self):
        return self.linop.H
    
    def _normal_linop(self):
        return self.linop.H * self.linop
    


"""
ReLU solver using ADMM. Implements Algorithm 3.1 of ADMM ReLU Paper.
Memory-efficient calculations of F and G with linops
@Daniel's implementation
"""
def admm_optimizer_linop(solver, X, y, max_iter, verbose=False):

    # Variables
    n, d = X.shape
    P_S = solver.P_S
    
    # Hyperplanes
    d_diags = get_hyperplane_cuts(X, solver.h)

    # Make efficient linear operators
    F = F_linop(d_diags, X)
    G = G_linop(d_diags, X)
    
    # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP in one long vector
    u = np.zeros((2, d, P_S))
    # v contrains v1 ... vP, w1 ... wP in one long vector
    v = np.zeros((2, d, P_S))
    # slacks s1 ... sP, t1 ... tP
    s = np.zeros((2, n, P_S))
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = np.zeros((2, d, P_S))
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = np.zeros((2, n, P_S))

    # Convert to vectorized style
    def tensor_to_vec(tensor):
        vec = np.array([])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[2]):
                vec = np.append(vec, tensor[i, :, j])
        return vec

    def vec_to_tensor(vec):
        tensor = np.zeros((2, d, P_S))
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[2]):
                inds = np.arange(d * j, d * (j + 1)) + i * d * P_S
                tensor[i, :, j] = vec[inds]
        return tensor

    # --------------- Precomputations ---------------
    start = time.perf_counter()
    F_matrix = np.hstack([np.hstack([d_diags[:,i, None] * X for i in range(P_S)]),
        np.hstack([-1 * d_diags[:,i, None] * X for i in range(P_S)])])
    A = np.eye(2 * d * P_S) + F_matrix.T @ F_matrix / solver.rho
    Glist = [(2 * d_diags[:, i, None] - 1) * X for i in range(P_S)]
    for i in range(2):
        for j in range(P_S):
            lower_ind = d * j + i * d * P_S
            upper_ind = d * (j+1) + i * d * P_S
            GiTGi = Glist[j].T @ Glist[j]
            A[lower_ind:upper_ind, lower_ind:upper_ind] += GiTGi
    L = LA.cholesky(A)
    b_1 = F.H * y / solver.rho
    stop = time.perf_counter()
    print(f'\nPre Computations Took {stop - start:.3f}s')

    # --------------- Iterative Updates ---------------
    # benchmark times
    time_u = 0
    time_v = 0
    time_s = 0

    # keep track of losses
    train_loss, train_acc = [], []

    k = 0 
    while k < max_iter:

        # keep track of losses
        y_hat = F * u
        train_loss.append(solver.loss_func(y_hat, y))
        train_acc.append(solver.acc_func(y_hat, y))
        if verbose: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # Solve linear system for u
        # TODO conjugate gradient intstead?
        start = time.perf_counter()
        b = tensor_to_vec(b_1 + v - lam + G.H * (s - nu))
        bhat = solve_triangular(L, b, lower=True)
        u = vec_to_tensor(solve_triangular(L.T, bhat, lower=False))
        stop = time.perf_counter()
        time_u += stop - start

        # upates on v = (v1...vP, w1...wP)
        start = time.perf_counter()
        v = np.maximum(1 - solver.beta / (solver.rho * LA.norm(u + lam, axis=1)[:, None, :]), 0) * (u + lam)
        stop = time.perf_counter()
        time_v += stop - start

        # updates on s = (s1...sP, t1...tP)
        start = time.perf_counter()
        s = np.maximum(G * u + nu, 0)
        stop = time.perf_counter()
        time_s += stop - start

        # finally, perform dual updates
        lam += (u - v) * solver.step / solver.rho
        nu += (G * u - s) * solver.step / solver.rho

        # iter step 
        k += 1        

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    v = tensor_to_vec(v)
    opt_weights = np.reshape(v, (P_S*2, d), order='C')
    solver.v = opt_weights[:P_S]
    solver.w = opt_weights[P_S:P_S*2]

    # collect metrics
    solver.metrics["train_loss"] = np.array(train_loss)
    solver.metrics["train_acc"] = np.array(train_acc)

    # Show times
    print(f'\nU updates Took {time_u:.3f}s')
    print(f'V updates Took {time_v:.3f}s')
    print(f'S updates Took {time_s:.3f}s')

    return solver
