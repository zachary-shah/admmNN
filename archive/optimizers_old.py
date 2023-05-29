
import numpy as np
import numpy.linalg as LA
import cvxpy as cp
import time

from utils.typing_utils import ArrayType, EvalFunction
from utils.admm_utils import ADMM_Params, FG_Operators, get_hyperplane_cuts, tensor_to_vec, proxl2
from utils.primal_update_utils import RBCD_update, ADMM_full_update, ADMM_cg_update


"""
ReLU solver using ADMM. Implements Algorithm 3.1 of ADMM ReLU Paper as subroutine.
"""
def admm_optimizer(parms: ADMM_Params,
                    X: np.ndarray, 
                    y: np.ndarray, 
                    loss_func: function,
                    acc_func: function,
                    max_iter: int, 
                    verbose: bool = False,
                    ):

    # --------------------- Setup ---------------------
    solver_metrics = {}
    n, d = X.shape
    P_S = parms.P_S
    
    # Hyperplanes (D_i samples)
    d_diags = get_hyperplane_cuts(X, P_S, seed=parms.seed)

    # utility operator to memory-efficient compute F*u and G*u
    OPS = FG_Operators(d_diags=d_diags, X=X)
    
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
    
    # compute A. TODO: parallelize FiFj and GjTGj computations
    # cholesky decomposition if not using conjugate gradient
    if not parms.admm_cg_solve:
        A = np.eye(2 * d * P_S)
        for i in range(P_S):
            for j in range(P_S):
                # perform multiplication 
                FiFj = OPS.F(i % P_S).T @ OPS.F(j % P_S) / parms.rho
                # assign to four quadrants
                A[i*d:(i+1)*d, j*d:(j+1)*d] += FiFj
                A[(i+P_S)*d:(i+P_S+1)*d, (j)*d:(j+1)*d] += - FiFj
                A[(i)*d:(i+1)*d, (j+P_S)*d:(j+P_S+1)*d] += - FiFj
                A[(i+P_S)*d:(i+P_S+1)*d, (j+P_S)*d:(j+P_S+1)*d] += FiFj
        for i in range(2):
            for j in range(P_S):
                lower_ind = d * j + i * d * P_S
                upper_ind = d * (j+1) + i * d * P_S
                A[lower_ind:upper_ind, lower_ind:upper_ind] += OPS.G(j).T @ OPS.G(j)

        b_1 = OPS.F_multop(y, transpose=True) / parms.rho

        L = LA.cholesky(A)
        del A

    stop = time.perf_counter()
    if verbose: print(f'\nPre Computations Took {stop - start:.3f}s')

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
        y_hat = OPS.F_multop(u)
        train_loss.append(loss_func(y_hat, y))
        train_acc.append(acc_func(y_hat, y))
        if verbose: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # updates on u = (u1...uP, z1....zP)
        # conjugate gradient solve
        start = time.perf_counter()
        if parms.admm_cg_solve:
            """
            @Daniel: Implement CG with preconditioners here
                 - use cg_params to specify stuff about preconditioners in admm_utils.ADMM_Params
            """
            raise NotImplementedError("Conjugate Gradient sovle for ADMM full step still to be impemented.")
        # else solve full linear system
        else:
            b = tensor_to_vec(b_1 + v - lam + OPS.G_multop(s - nu, transpose=True))
            bhat = solve_triangular(L, b, lower=True)
            u = vec_to_tensor(solve_triangular(L.T, bhat, lower=False))

        stop = time.perf_counter()
        time_u += stop - start

        # upates on v = (v1...vP, w1...wP)
        start = time.perf_counter()
        v = np.maximum(1 - parms.beta / (parms.rho * LA.norm(u + lam, axis=1)[:, None, :]), 0) * (u + lam)
        stop = time.perf_counter()
        time_v += stop - start

        # updates on s = (s1...sP, t1...tP)
        start = time.perf_counter()
        s = np.maximum(OPS.G_multop(u) + nu, 0)
        stop = time.perf_counter()
        time_s += stop - start

        # finally, perform dual updates
        lam += (u - v) * parms.step / parms.rho
        nu += (OPS.G_multop(u) - s) * parms.step / parms.rho

        # iter step 
        k += 1        

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    v = tensor_to_vec(v)
    opt_weights = np.reshape(v, (P_S*2, d), order='C')
    v = opt_weights[:P_S]
    w = opt_weights[P_S:P_S*2]

    # collect metrics
    solver_metrics["train_loss"] = np.array(train_loss)
    solver_metrics["train_acc"] = np.array(train_acc)

    # Show times
    if verbose:
        print(f'\nU updates Took {time_u:.3f}s')
        print(f'V updates Took {time_v:.3f}s')
        print(f'S updates Took {time_s:.3f}s')

    return v, w, solver_metrics


"""
ADMM with RBCD updates
@Miria implementation made to fit with our framework
"""
def admm_rbcd_optimizer(parms: ADMM_Params,
                    X: np.ndarray, 
                    y: np.ndarray, 
                    loss_func: function,
                    acc_func: function,
                    max_iter: int, 
                    verbose: bool = False,
                    ):
    
    # --------------------- Setup ---------------------
    solver_metrics = {}
    n, d = X.shape
    P_S = parms.P_S
    
    # Hyperplanes
    d_diags = get_hyperplane_cuts(X, P_S)

    # utility operator to memory-efficient compute F*u and G*u
    OPS = FG_Operators(d_diags=d_diags, X=X)
    
    # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP 
    u = np.zeros((2, d, P_S))
    # v contrains v1 ... vP, w1 ... wP
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

    # --------------- Precomputations ---------------
    start = time.perf_counter()
    # compute Xi.T @ X only for this 
    GiTGi = X.T @ X
    y = y.squeeze()
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

        # ----------- METRIC COMPUTATIONS -----------------
        y_hat = OPS.F_multop(u)
        train_loss.append(loss_func(y_hat, y))
        train_acc.append(acc_func(y_hat, y))
        if verbose: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # ----------- PERFORM RBCD UPDATE -----------------
        start = time.perf_counter()
        parms, u = RBCD_update(parms, OPS, y, y_hat, u, v, s, nu, lam, GiTGi, loss_func, verbose=verbose)
        time_u += time.perf_counter() - start

        # parameter updates
        parms.RBCD_thresh *= parms.RBCD_thresh_decay
        parms.rho += parms.rho_increment
        parms.gamma_ratio *= parms.gamma_ratio_decay

        # ----------- OTHER PARAMETER UPDATES -----------------
        # upates on v = (v1...vP, w1...wP) via prox operator
        start = time.perf_counter()
        v = np.maximum(1 - parms.beta / (parms.rho * LA.norm(u + lam, axis=1)[:, None, :]), 0) * (u + lam)
        time_v += time.perf_counter() - start

        # updates on s = (s1...sP, t1...tP)
        start = time.perf_counter()
        s = np.maximum(OPS.G_multop(u) + nu, 0)
        time_s += time.perf_counter() - start

        # finally, perform dual updates
        lam += (u - v) * parms.gamma_ratio
        nu += (OPS.G_multop(u) - s) * parms.gamma_ratio

        # iter step 
        k += 1        

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    v = tensor_to_vec(v)
    opt_weights = np.reshape(v, (P_S*2, d), order='C')
    v = opt_weights[:P_S]
    w = opt_weights[P_S:P_S*2]

    # collect metrics
    solver_metrics["train_loss"] = np.array(train_loss)
    solver_metrics["train_acc"] = np.array(train_acc)

    # Show times
    print(f'\nU updates Took {time_u:.3f}s')
    print(f'V updates Took {time_v:.3f}s')
    print(f'S updates Took {time_s:.3f}s')

    return v, w, solver_metrics

