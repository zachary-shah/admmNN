"""
Implementation of our optimizers
"""

import numpy as np
import numpy.linalg as LA
import cvxpy as cp
from time import perf_counter
from typing import Tuple, Union

from utils.typing_utils import ArrayType, EvalFunction, convert_backend_type
from utils.admm_utils import ADMM_Params, FG_Operators, Linear_Sys, get_hyperplane_cuts, tensor_to_vec, proxl2
from utils.primal_update_utils import RBCD_update
from utils.relu_utils import optimal_weights_transform

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
                    max_time: int = 120,
                    val_data: Union[None, Tuple[ArrayType, ArrayType]] = None,
                    verbose: bool = False,
                    ):
    
    if verbose: print(f"Beginning optimization! Mode: {parms.mode}")

    # add validation if desired
    validate = False
    if val_data is not None:
        validate = True
        X_val, y_val = val_data

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

    # add validation metrics if provided
    if validate:

        alpha, u = optimal_weights_transform(v, w, P_S, d, verbose=verbose)
        y_hat_val = mnp.relu(X_val @ u) @ alpha

        y_hat = (F @ (v.value - w.value))
        solver_metrics["val_loss"] = mnp.array([loss_func(y_hat_val, y_val)] * max_iter)
        solver_metrics["val_acc"] = mnp.array([acc_func(y_hat_val, y_val)] * max_iter)

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
                    max_time: int = 120,
                    val_data: Union[None, Tuple[ArrayType, ArrayType]] = None,
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
    if verbose: print(f"\td_diags.shape: {d_diags.shape}")

    # utility operator to memory-efficient compute F*u and G*u
    OPS = FG_Operators(d_diags=d_diags, X=X, rho=parms.rho, mem_save=parms.memory_save)

    # get validation data if provided
    validate = False
    if val_data is not None:
        X_val, y_val = val_data
        validate = True

    # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP 
    u = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend, device=parms.device)
    # v contrains v1 ... vP, w1 ... wP
    v = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend, device=parms.device)
    # slacks s1 ... sP, t1 ... tP
    s = mnp.zeros((2, n, P_S), backend_type=parms.datatype_backend, device=parms.device)
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = mnp.zeros((2, d, P_S), backend_type=parms.datatype_backend, device=parms.device)
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = mnp.zeros((2, n, P_S), backend_type=parms.datatype_backend, device=parms.device)

    # --------------- Precomputations ---------------
    # get time that has passed each iteration (first iteration has a lot of time due to precomputation)
    start = perf_counter()
    iteration_time = 0
    total_time = 0

    if verbose: print("  Completing precomputations...")

    if parms.mode == "ADMM":
        # do precomputations in initialization of the linear system
        ls = Linear_Sys(OPS=OPS, 
                        params=parms,
                        verbose=verbose)
        
        b_1 = OPS.F_multop(y, transpose=True) / parms.rho

    elif parms.mode == "ADMM-RBCD":
        # compute Xi.T @ X only for this 
        GiTGi = X.T @ X
        y = y.squeeze()

    else:
        raise NotImplementedError("Unexpected mode for ADMM optimization.")
    
    time_precomp = perf_counter() - start
    iteration_time += time_precomp

    if verbose: print(f'\tPre Computations Took {time_precomp:.3f}s')

    # --------------- Iterative Updates ---------------
    if verbose: print(f'\nBeginning descent with maximum {max_iter} iterations and max solve time of {max_time}: ')

    # benchmark times
    time_u, time_v, time_s, time_dual = 0, 0, 0, 0

    # keep track of losses
    train_loss, train_acc, iteration_timepoints = [], [], []
    if validate: val_loss, val_acc = [], []

    # optimality conditions
    u_v_dist = mnp.inf(backend_type=parms.datatype_backend)
    u_optimality = mnp.inf(backend_type=parms.datatype_backend)
    v_optimality = mnp.inf(backend_type=parms.datatype_backend)
    k = 1 

    # optimal if primal and dual conditions all within tolerance
    def check_optimal():
        not_optimal = True
        tol = parms.optimality_tolerance
        if u_v_dist <= tol and u_optimality <= tol and v_optimality <= tol:
            not_optimal = False
        return not_optimal

    while check_optimal():

        # ----------- PERFORM U UPDATE -----------------
        start = perf_counter()

        # admm full step
        if parms.mode == "ADMM":
            b = b_1 + v - lam + OPS.G_multop(s - nu, transpose=True)
            u = ls.solve(b)
        # rbcd steps
        elif parms.mode == "ADMM-RBCD":
            parms, u = RBCD_update(parms, OPS, y, u, v, s, nu, lam, GiTGi, loss_func, verbose=verbose)
            # parameter updates
            parms.RBCD_thresh *= parms.RBCD_thresh_decay
            parms.gamma_ratio *= parms.gamma_ratio_decay
            parms.rho += parms.rho_increment
            
        time_u += perf_counter() - start
        iteration_time += perf_counter() - start

        # ----------- OTHER PARAMETER UPDATES -----------------
        # upates on v = (v1...vP, w1...wP) via prox operator
        start = perf_counter()
        if parms.datatype_backend == "jax":
            # v update
            v = v.at[0].set(proxl2(u[0] + lam[0], beta=parms.beta, gamma=1 / parms.rho))
            # w update
            v = v.at[1].set(proxl2(u[1] + lam[1], beta=parms.beta, gamma=1 / parms.rho))
        else:
            # v, w update
            v[0] = proxl2(u[0] + lam[0], beta=parms.beta, gamma=1 / parms.rho)
            v[1] = proxl2(u[1] + lam[1], beta=parms.beta, gamma=1 / parms.rho)
        time_v += perf_counter() - start
        iteration_time += perf_counter() - start

        # updates on s = (s1...sP, t1...tP)
        start = perf_counter()
        Gu = OPS.G_multop(u)
        s = mnp.relu(Gu + nu)
        time_s += perf_counter() - start
        iteration_time += perf_counter() - start

        # finally, perform dual updates on lam=(lam11...lam2P), nu=(nu11...nu2P)
        start = perf_counter()
        lam += (u - v) * parms.gamma_ratio
        nu += (Gu - s) * parms.gamma_ratio
        time_dual += perf_counter() - start
        iteration_time += perf_counter() - start

        # calculations for checking optimality conditions
        y_hat = OPS.F_multop(u)
        u_v_dist = mnp.norm(u - v) + mnp.norm(Gu - s)
        u_optimality = mnp.norm(OPS.F_multop(y_hat - y.squeeze(), transpose=True) + parms.rho * (lam + OPS.G_multop(nu, transpose=True)))
        v_optimality = mnp.norm(parms.beta * v / mnp.norm(v, axis=2, keepdims=True) - parms.rho * lam)
        if verbose: print(f"iter: {k}\n  u-v dist = {u_v_dist}, u resid = {u_optimality}, v resid = {v_optimality}")

        # ----------- METRIC COMPUTATIONS -----------------
        train_loss.append(convert_backend_type(loss_func(y_hat, y), target_backend="numpy")) 
        train_acc.append(convert_backend_type(acc_func(y_hat, y), target_backend="numpy"))
        if validate:
            u_transform, alpha_transform = optimal_weights_transform(v[0], v[1], P_S, d, verbose=verbose)
            if len(alpha_transform) > 0:
                y_hat_val = mnp.relu(X_val @ u_transform) @ alpha_transform
                # loss and accuracy calculation
                val_loss.append(convert_backend_type(loss_func(y_hat_val, y_val), target_backend="numpy"))
                val_acc.append(convert_backend_type(acc_func(y_hat_val, y_val), target_backend="numpy"))
            # handle case where no weights are non-zero
            else:
                val_loss.append(mnp.inf())
                val_acc.append(0)

            if verbose: print(f"  tr_loss = {train_loss[-1]}, tr_acc = {train_acc[-1]}, val_acc = {val_acc[-1]}")
        elif verbose: print(f"  loss = {train_loss[-1]}, acc = {train_acc[-1]}")

        # keep track of iteration times
        total_time += iteration_time
        iteration_timepoints.append(total_time)
        iteration_time = 0

        if total_time > max_time:
            if verbose: print(f"Warning: Solve time ({total_time}s) has exceeded max time of {max_time}s. Optimization not guranteed.")
            break

        if k == max_iter:
            if verbose: print(f"Warning: Reached max iteration count of {k}. Optimization not guranteed.")
            break

        # iter step 
        k += 1        

    # collect metrics (just keep as numpy arrays by default)
    solver_metrics["iteration_timepoints"] = mnp.array(iteration_timepoints)
    solver_metrics["train_loss"] = mnp.array(train_loss)
    solver_metrics["train_acc"] = mnp.array(train_acc)

    solver_metrics["solve_time_breakdown"] = dict(
        total_time=total_time,
        time_precomp=time_precomp,
        time_u=time_u,
        time_v=time_v,
        time_s=time_s,
        time_dual=time_dual,
    )

    if validate:
        solver_metrics["val_loss"] = mnp.array(val_loss)
        solver_metrics["val_acc"] = mnp.array(val_acc)

    # Show times
    if verbose:
        print(f"""\nOptimization runner terminating.\
        \nMetrics summary:\
        \n\tFinal train loss: {train_loss[-1]}\
        \n\tFinal train accuracy: {train_acc[-1]}\
        \nComputation times summary:\
        \n\tTotal solve time: {total_time:.4f}s\
        \n\tPrecomputations:    {time_precomp:.4f}s\
        \n\tTotal U updates:    {time_u:.4f}s\
        \n\tTotal V updates:    {time_v:.4f}s\
        \n\tTotal S updates:    {time_s:.4f}s\
        \n\tTotal Dual updates: {time_dual:.4f}s""")

    # Optimal Weights v1...vP_S w1...wP_S of C-ReLU Problem
    return v[0], v[1], solver_metrics


## FOR CHECKING THAT OPTIMIZERS ARE IMPLEMENTED 
IMPLEMENTED_OPTIMIZERS = [cvxpy_optimizer, admm_optimizer]