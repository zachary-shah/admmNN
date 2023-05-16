"""
Implementation of our optimizers
"""

import numpy as np
import numpy.linalg as LA
import cvxpy as cp
import sigpy as sp
import time
from scipy.linalg import block_diag, solve_triangular
from linops import F_linop, G_linop

from relu_utils import get_hyperplane_cuts

IMPLEMENTED_OPTIMIZERS = ["cvxpy", "admm"]

"""
Solve optimizaton problem via cvxpy 
"""
def cvxpy_optimizer(solver, X, y, max_iter, verbose=False):
    # Variables
    d = solver.d + int(solver.bias)
    P_S = solver.P_S

    v = cp.Variable((d * P_S))
    w = cp.Variable((d * P_S))
    
    d_diags = get_hyperplane_cuts(X, solver.h)

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

        return l2_term + solver.beta * group_sparsity
    
    # Solve via cvxpy
    prob = cp.Problem(cp.Minimize(obj_func(F, y, v, w)),
                    [((2 * d_diags[:, i, None] - 1) * X) @ v[i*d:(i+1)*d] >= 0 for i in range(P_S)] + \
                    [((2 * d_diags[:, i, None] - 1) * X) @ w[i*d:(i+1)*d] >= 0 for i in range(P_S)])
    prob.solve(verbose=verbose, solver='ECOS')

    stats = prob.solver_stats
    
    # Grab optimal values 
    solver.v = np.reshape(v.value, (P_S, d), order='C')
    solver.w = np.reshape(w.value, (P_S, d), order='C')

    # metrics
    solver.metrics["train_loss"] = np.array([0] * max_iter)
    solver.metrics["train_acc"] = np.array([0] * max_iter)

    return solver

# def apply_G(vec):


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
    print(f'Pre Computations Took {stop - start:.3f}s')

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
    print(f'Solving Times:')
    print(f'U updates Took {time_u:.3f}s')
    print(f'V updates Took {time_v:.3f}s')
    print(f'S updates Took {time_s:.3f}s')

    return solver


"""
ReLU solver using ADMM. Implements Algorithm 3.1 of ADMM ReLU Paper.
"""
def admm_optimizer(solver, X, y, max_iter, verbose=False):


    # Variables
    n, d = X.shape
    P_S = solver.P_S
    
    d_diags = get_hyperplane_cuts(X, solver.h)

    # F here is n x (2d*P_S)
    F = np.hstack([np.hstack([d_diags[:,i, None] * X for i in range(P_S)]),
        np.hstack([-1 * d_diags[:,i, None] * X for i in range(P_S)])])

    # G is block diagonal 2*n*P_S x 2*d*P_s
    Glist = [(2 * d_diags[:, i, None] - 1) * X for i in range(P_S)]
    G = block_diag(*Glist * 2)

    

    ### INITIALIZATIONS OF OPTIMIZATION VARIABLES 

    # u contains u1 ... uP, z1... zP in one long vector
    u = np.zeros((2 * d * P_S, 1))
    # v contrains v1 ... vP, w1 ... wP in one long vector
    v = np.zeros((2 * d * P_S, 1))

    # slacks s1 ... sP, t1 ... tP
    s = np.zeros((2 * n * P_S, 1))

    # dual variables
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = np.zeros((2 * d * P_S, 1))
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = np.zeros((2 * n * P_S, 1))

    ### PRECOMPUTATIONS

    # for u update: A in 2dP_s x 2dP_s
    A = np.eye(2*d*P_S) + F.T @ F / solver.rho + G.T @ G

    # cholesky factorization
    L = LA.cholesky(A)
    # extra precompute for u update step
    b_1 = F.T @ y / solver.rho

    ## ITERATIVE UPDATES 
    time_u = 0
    time_v = 0
    time_s = 0

    # keep track of losses
    train_loss, train_acc = [], []

    k = 0 
    while k < max_iter:

        # keep track of losses
        y_hat = F @ u
        train_loss.append(solver.loss_func(y_hat, y))
        train_acc.append(solver.acc_func(y_hat, y))
        if verbose and k%10 == 0: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # first, conduct the primal update on u (u1...uP, z1...zP)
        start = time.perf_counter()
        b = b_1 + v - lam + G.T @ (s - nu)
        bhat = solve_triangular(L, b, lower=True)
        u = solve_triangular(L.T, bhat, lower=False)
        stop = time.perf_counter()
        time_u += stop - start

        # second, perform updates of v and s (TODO: parallelize v and s updates)
        # upates on v = (v1...vP, w1...wP)
        start = time.perf_counter()
        for i in range(2 * P_S):
            inds = np.arange(d*i, d*(i+1))
            if True: #not np.isclose(solver.rho * LA.norm(u[inds] + lam[inds]), 0):
                v[inds] = np.maximum(1 - solver.beta/(solver.rho * LA.norm(u[inds] + lam[inds])), 0) * (u[inds] + lam[inds])
            else:
                v[inds] = 0
        stop = time.perf_counter()
        time_v += stop - start

        # updates on s = (s1...sP, t1...tP)
        start = time.perf_counter()
        for i in range(2 * P_S):
            s[i*n:(i+1)*n] = np.maximum(Glist[i % P_S] @ u[i*d:(i+1)*d] + nu[i*n:(i+1)*n], 0)
        stop = time.perf_counter()
        time_s += stop - start

        # finally, perform dual updates
        lam += solver.step / solver.rho * (u - v)
        nu += solver.step / solver.rho * (G @ u - s)

        # iter step 
        k += 1        

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    opt_weights = np.reshape(v, (P_S*2, d), order='C')
    solver.v = opt_weights[:P_S]
    solver.w = opt_weights[P_S:P_S*2]

    # collect metrics
    solver.metrics["train_loss"] = np.array(train_loss)
    solver.metrics["train_acc"] = np.array(train_acc)

    # Show times
    print(f'Solving Times:')
    print(f'U update = {time_u:.3f}')
    print(f'V update = {time_v:.3f}')
    print(f'S update = {time_s:.3f}')

    return solver