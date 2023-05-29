"""
Implementation of our optimizers
"""

import numpy as np
import numpy.linalg as LA
import cvxpy as cp
from time import perf_counter

from utils.typing_utils import ArrayType, EvalFunction
from utils.admm_utils import ADMM_Params, FG_Operators, get_hyperplane_cuts, tensor_to_vec, proxl2
from utils.primal_update_utils import RBCD_update, ADMM_full_update, ADMM_cg_update

import utils.math_utils as mnp

"""
Solve optimizaton problem via cvxpy 
"""
def cvxpy_optimizer(parms: ADMM_Params,
                    X: ArrayType, 
                    y: ArrayType, 
                    loss_func: EvalFunction,
                    acc_func: EvalFunction,
                    max_iter: int, 
                    verbose: bool = False,
                    ):
    
    if verbose: print(f"Beginning optimization! Mode: {parms.mode}")

    # Setup / Variables
    solver_metrics = {}
    n, d = X.shape
    P_S = parms.P_S

    v = cp.Variable((d * P_S))
    w = cp.Variable((d * P_S))
    
    d_diags = get_hyperplane_cuts(X, P_S, seed=parms.seed)

    # Construct all possible data enumerations (n x P_S * d)
    F = mnp.hstack([d_diags[:, i, None] * X for i in range(P_S)])

    # Objective Function
    def obj_func(F, y, v, w):
        l2_term = cp.sum_squares(
            (F @ (v - w)) - y[:, 0]
        )
        group_sparsity = 0
        for i in range(P_S):
            group_sparsity += cp.norm2(v[i*d:(i+1)*d])
            group_sparsity += cp.norm2(w[i*d:(i+1)*d])

        return l2_term + parms.beta * group_sparsity
    
    # Solve via cvxpy
    prob = cp.Problem(cp.Minimize(obj_func(F, y, v, w)),
                    [((2 * d_diags[:, i, None] - 1) * X) @ v[i*d:(i+1)*d] >= 0 for i in range(P_S)] + \
                    [((2 * d_diags[:, i, None] - 1) * X) @ w[i*d:(i+1)*d] >= 0 for i in range(P_S)])
    prob.solve(verbose=verbose, solver='ECOS')
    
    # Grab optimal values 
    v = mnp.reshape(v.value, (P_S, d))
    w = mnp.reshape(w.value, (P_S, d))

    # solve metrics
    y_hat = (F @ (v.value - w.value))
    solver_metrics["train_loss"] = mnp.array([loss_func(y_hat, y)] * max_iter)
    solver_metrics["train_acc"] = mnp.array([acc_func(y_hat, y)] * max_iter)

    return v, w, solver_metrics


"""
One solver to perform ADMM with either ADMM or RBCD updates
"""
def admm_optimizer(parms: ADMM_Params,
                    X: ArrayType, 
                    y: ArrayType, 
                    loss_func: EvalFunction,
                    acc_func: EvalFunction,
                    max_iter: int, 
                    verbose: bool = False,
                    ):
    
    # --------------------- Setup ---------------------
    solver_metrics = {}
    n, d = X.shape
    P_S = parms.P_S

    if verbose: print(f"\nBeginning optimization! Mode: {parms.mode}")

    # Hyperplanes
    if verbose: print("  Sampling hyperplane cuts (D_h matrices)...")
    d_diags = get_hyperplane_cuts(X, P_S, seed=parms.seed)
    print(f"\td_diags.shape: {d_diags.shape}")

    # utility operator to memory-efficient compute F*u and G*u
    OPS = FG_Operators(d_diags=d_diags, X=X)
    
    # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP 
    u = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend)
    # v contrains v1 ... vP, w1 ... wP
    v = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend)
    # slacks s1 ... sP, t1 ... tP
    s = mnp.zeros((2, n, P_S), backend_type=parms.datatype_backend)
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend)
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = mnp.zeros((2, n, P_S), backend_type=parms.datatype_backend)

    # --------------- Precomputations ---------------
    start = perf_counter()

    if verbose: print("  Completing precomputations...")

    if parms.mode == "ADMM":

        A = mnp.eye(2 * d * P_S, device=parms.device, backend_type=parms.datatype_backend)
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

        # cholesky decomposition if not using conjugate gradient
        if not parms.admm_cg_solve:
            L = LA.cholesky(A)
            del A

    elif parms.mode == "ADMM-RBCD":
        # compute Xi.T @ X only for this 
        GiTGi = X.T @ X
        y = y.squeeze()
    else:
        raise NotImplementedError("Unexpected mode for ADMM optimization.")
    
    time_precomp = perf_counter() - start
    
    if verbose: print(f'\tPre Computations Took {time_precomp:.3f}s')

    # --------------- Iterative Updates ---------------
    if verbose: print(f'\nBeginning descent with maximum {max_iter} iterations: ')

    # benchmark times
    time_u, time_v, time_s, time_dual = 0, 0, 0, 0

    # keep track of losses
    train_loss, train_acc = [], []

    k = 0 
    while k < max_iter:

        # ----------- METRIC COMPUTATIONS -----------------
        y_hat = OPS.F_multop(u)
        train_loss.append(loss_func(y_hat, y))
        train_acc.append(acc_func(y_hat, y))
        if verbose: print(f"iter = {k}, loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # ----------- PERFORM U UPDATE -----------------
        start = perf_counter()

        # admm full step
        if parms.mode == "ADMM":
            # solve linear system approximately with conjugate gradient
            if parms.admm_cg_solve:
                u = ADMM_cg_update(parms, OPS, v, s, nu, lam, A, b_1)
            # else solve full linear system
            else:
                u = ADMM_full_update(parms, OPS, v, s, nu, lam, L, b_1)
        # rbcd steps
        elif parms.mode == "ADMM-RBCD":
            parms, u = RBCD_update(parms, OPS, y, y_hat, u, v, s, nu, lam, GiTGi, loss_func, verbose=verbose)
            # parameter updates
            parms.RBCD_thresh *= parms.RBCD_thresh_decay
            parms.gamma_ratio *= parms.gamma_ratio_decay
            parms.rho += parms.rho_increment
            
        time_u += perf_counter() - start

        # ----------- OTHER PARAMETER UPDATES -----------------
        # upates on v = (v1...vP, w1...wP) via prox operator
        start = perf_counter()
        if parms.mode == "ADMM":
            v = mnp.relu(1 - parms.beta / (parms.rho * mnp.norm(u + lam, axis=1)[:, None, :])) * (u + lam)
        else:
            # v update
            v[0] = proxl2(u[0] + lam[0], beta=parms.beta, gamma=1 / parms.rho)
            # w update
            v[1] = proxl2(u[1] + lam[1], beta=parms.beta, gamma=1 / parms.rho)

        time_v += perf_counter() - start

        # updates on s = (s1...sP, t1...tP)
        start = perf_counter()
        Gu = OPS.G_multop(u)
        s = mnp.relu(Gu + nu)
        time_s += perf_counter() - start

        # finally, perform dual updates on lam=(lam11...lam2P), nu=(nu11...nu2P)
        start = perf_counter()
        lam += (u - v) * parms.gamma_ratio
        nu += (Gu - s) * parms.gamma_ratio
        time_dual += perf_counter() - start

        # iter step 
        k += 1        

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    v = tensor_to_vec(v)
    opt_weights = mnp.reshape(v, (P_S*2, d))
    v = opt_weights[:P_S]
    w = opt_weights[P_S:P_S*2]

    # collect metrics (just keep as numpy arrays by default)
    solver_metrics["train_loss"] = mnp.array(train_loss)
    solver_metrics["train_acc"] = mnp.array(train_acc)

    # Show times
    if verbose:
        print(f"""\nOptimization complete!\
        \nMetrics summary:\
        \n\tFinal loss: {train_loss[-1]}\
        \n\tFinal accuracy: {train_acc[-1]}\
        \nComputation times summary:\
        \n\tPrecomputations:    {time_precomp:.4f}s\
        \n\tTotal U updates:    {time_u:.4f}s\
        \n\tTotal V updates:    {time_v:.4f}s\
        \n\tTotal S updates:    {time_s:.4f}s\
        \n\tTotal Dual updates: {time_dual:.4f}s""")

    return v, w, solver_metrics


## FOR CHECKING THAT OPTIMIZERS ARE IMPLEMENTED 
IMPLEMENTED_OPTIMIZERS = [cvxpy_optimizer, admm_optimizer]