"""
Main implementations for u updates during ADMM / ADMM-RBCD
"""

from math import ceil
from typing import Tuple

from utils.typing_utils import ArrayType, EvalFunction
from utils.admm_utils import FG_Operators, ADMM_Params, vec_to_tensor, tensor_to_vec
import utils.math_utils as mnp

# """
# Performs RBCD updates step for ADMM-RBCD descent
# """
# def RBCD_update(parms: ADMM_Params, 
#                 OPS: FG_Operators, 
#                 y: ArrayType, 
#                 y_hat: ArrayType, 
#                 u: ArrayType, 
#                 v: ArrayType, 
#                 s: ArrayType, 
#                 nu: ArrayType, 
#                 lam: ArrayType, 
#                 GiTGi: ArrayType, 
#                 loss_func: EvalFunction,
#                 verbose: bool = False) -> Tuple[ADMM_Params, ArrayType]:
    
#     if verbose: print("  Beginning RBCD update...")

#     e_diags = 2 * OPS.d_diags - 1
    
#     # Initialize objective
#     y_hat = mnp.sum((OPS.d_diags * (OPS.X @ (u[0] - u[1]))), axis=1)

#     stil = OPS.X.T @ (e_diags * (s - nu))
#     #stil = OPS.G_multop(s - nu, transpose=True)

#     dcosts = mnp.ones(ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize))) * 1e8
#     ptr, k = 0, 0  # k is current count of iterations
#     while dcosts.mean() > parms.RBCD_thresh:
#         k += 1
#         i = mnp.random_choice(parms.P_S, size=parms.RBCD_blocksize, replace=False)

#         # Calculate training loss via u and z (without regularization) and get gradients
#         loss1 = loss_func(y_hat, y)

#         if parms.loss_type == 'mse':
#             grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (y_hat - y)[:,None])
#         elif parms.loss_type == 'ce':
#             grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (-2 * y + 2 / (1 + mnp.exp(-2 * y_hat)))[:,None])

#         grad2u = u[0][:, i] - v[0][:, i] + lam[0][:, i] + GiTGi @ u[0][:, i] - stil[0][:, i]
#         grad2z = u[1][:, i] - v[1][:, i] + lam[1][:, i] + GiTGi @ u[1][:, i] - stil[1][:, i]
#         gradu = grad1 + parms.rho * grad2u
#         gradz = -grad1 + parms.rho * grad2z

#         # ----------- Determine the step size using line search -----------------
#         alpha = parms.alpha0
#         while True:  # Emulate a do-while loop
#             du = -alpha * gradu
#             dz = -alpha * gradz

#             # Current prediction (via convex formulation)
#             yhat_new = y_hat + mnp.sum(OPS.d_diags[:, i] * (OPS.X @ (du - dz)), axis=1)
            
#             # compute loss
#             dloss = loss_func(yhat_new, y) - loss1

#             ddist1 = ((u[0][:, i] + du - v[0][:, i] + lam[0][:, i]) ** 2).sum() + \
#                     ((u[1][:, i] + dz - v[1][:, i] + lam[1][:, i]) ** 2).sum() - \
#                     ((u[0][:, i] - v[0][:, i] + lam[0][:, i]) ** 2).sum() + \
#                     ((u[1][:, i] - v[1][:, i] + lam[1][:, i]) ** 2).sum()

#             ddist2 = ((e_diags[:, i] * (OPS.X @ (u[0][:, i] + du)) - s[0][:, i] + nu[0][:, i]) ** 2).sum() + \
#                     ((e_diags[:, i] * (OPS.X @ (u[1][:, i] + dz)) - s[1][:, i] + nu[1][:, i]) ** 2).sum() - \
#                     ((e_diags[:, i] * (OPS.X @ u[0][:, i]) - s[0][:, i] + nu[0][:, i]) ** 2).sum() - \
#                     ((e_diags[:, i] * (OPS.X @ u[1][:, i]) - s[1][:, i] + nu[1][:, i]) ** 2).sum()
#             dcost = dloss + (ddist1 + ddist2) * parms.rho / 2

#             # Armijo's rule
#             if alpha <= 1e-8 or dcost <= -1e-3 * mnp.sqrt((du ** 2).sum() + (dz ** 2).sum()):
#                 break
#             alpha /= 2.5
#             # Decaying basic step size
#             parms.alpha0 = mnp.maximum(1e-10, parms.alpha0 / 1.5)

#         # Update u, z, and objective
#         u[0][:, i] += du
#         u[1][:, i] += dz
#         dcosts[ptr] = -dcost
#         # Update circular buffer
#         ptr = (ptr + 1) % ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize))
#         parms.alpha0 *= 1.05
#         if verbose and k % 20 == 0:
#             print('      Iteration', k, ', alpha:', alpha, ', delta:', dcosts.mean().item())

#     return parms, u


"""
Performs RBCD updates step for ADMM-RBCD descent
"""
def RBCD_update(parms: ADMM_Params, 
                OPS: FG_Operators, 
                y: ArrayType, 
                y_hat: ArrayType, 
                u: ArrayType, 
                v: ArrayType, 
                s: ArrayType, 
                nu: ArrayType, 
                lam: ArrayType, 
                GiTGi: ArrayType, 
                loss_func: EvalFunction,
                verbose: bool = False) -> Tuple[ADMM_Params, ArrayType]:
    
    if verbose: print(f"  Beginning RBCD update...\n\tloss = {parms.loss_type}, \n\tloss_func = {loss_func}")

    emat = 2 * OPS.d_diags - 1
    dmat = OPS.d_diags

    u_orig = u.copy()
    v_orig = v.copy()
    s_orig = s.copy()
    lam_orig = lam.copy()
    nu_orig = nu.copy()

    u, z = u_orig[0], u_orig[1]
    v, w = v_orig[0], v_orig[1]
    s, t = s_orig[0], s_orig[1]
    lam1, lam2 = lam_orig[0], lam_orig[1]
    nu1, nu2 = nu_orig[0], nu_orig[1]
    
    # Initialize objective
    stil = OPS.X.T @ (emat * (s - nu1))
    ttil = OPS.X.T @ (emat * (t - nu2))
    y_hat = mnp.sum(dmat * (OPS.X @ (u - z)), axis=1) 

    dcosts = mnp.ones(ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize)), backend_type=parms.datatype_backend) * 1e8
    ptr, k = 0, 0  # k is current count of iterations
    while dcosts.mean() > parms.RBCD_thresh:
        k += 1
        i = mnp.random_choice(parms.P_S, size=parms.RBCD_blocksize, replace=False, backend_type=parms.datatype_backend)

        # Calculate training loss via u and z (without regularization) and get gradients
        loss1 = loss_func(y_hat, y)

        if parms.loss_type == 'mse':
            grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (y_hat - y)[:,None])
        elif parms.loss_type == 'ce':
            grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (-2 * y + 2 / (1 + mnp.exp(-2 * y_hat)))[:,None])

        grad2u = u[:, i] - v[:, i] + lam1[:, i] + GiTGi @ u[:, i] - stil[:, i]
        grad2z = z[:, i] - w[:, i] + lam2[:, i] + GiTGi @ z[:, i] - ttil[:, i]
        gradu = grad1 + parms.rho * grad2u
        gradz = -grad1 + parms.rho * grad2z

        # ----------- Determine the step size using line search -----------------
        alpha = parms.alpha0
        while True:  # Emulate a do-while loop
            du = -alpha * gradu
            dz = -alpha * gradz

            # Current prediction (via convex formulation)
            yhat_new = y_hat + mnp.sum(dmat[:, i] * (OPS.X @ (du - dz)), axis=1)

            dloss = loss_func(yhat_new, y) - loss1

            ddist1 = mnp.sum((u[:, i] + du - v[:, i] + lam1[:, i]) ** 2) + \
                     mnp.sum((z[:, i] + dz - w[:, i] + lam2[:, i]) ** 2) - \
                     mnp.sum((u[:, i] - v[:, i] + lam1[:, i]) ** 2) + \
                     mnp.sum((z[:, i] - w[:, i] + lam2[:, i]) ** 2)
            ddist2 = mnp.sum((emat[:, i] * (OPS.X @ (u[:, i] + du)) - s[:, i] + nu1[:, i]) ** 2) + \
                     mnp.sum((emat[:, i] * (OPS.X @ (z[:, i] + dz)) - t[:, i] + nu2[:, i]) ** 2) - \
                     mnp.sum((emat[:, i] * (OPS.X @ u[:, i]) - s[:, i] + nu1[:, i]) ** 2) - \
                     mnp.sum((emat[:, i] * (OPS.X @ z[:, i]) - t[:, i] + nu2[:, i]) ** 2)
            dcost = dloss + (ddist1 + ddist2) * parms.rho / 2

            # Armijo's rule
            if alpha <= 1e-8 or dcost <= -1e-3 * mnp.sqrt(mnp.sum((du ** 2)) + mnp.sum((dz ** 2))):
                break
            alpha /= 2.5
            # Decaying basic step size
            parms.alpha0 = mnp.maximum(1e-10, parms.alpha0 / 1.5)

        # Update u, z, and objective
        u[:, i] += du
        z[:, i] += dz
        dcosts[ptr] = -dcost
        # Update circular buffer
        ptr = (ptr + 1) % ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize))
        parms.alpha0 *= 1.05
        if verbose and k % 20 == 0:
            print('\tIteration', k, ', alpha:', alpha, ', delta:', dcosts.mean())

    # concatenate
    u_out = mnp.zeros_like(u_orig)
    u_out[0] = u.copy()
    u_out[1] = z.copy()

    return parms, u_out.copy()



"""
Performs Full ADMM update step using precomputed cholesky for linear systems solve
"""
def ADMM_full_update(parms: ADMM_Params, 
                     OPS: FG_Operators, 
                     v: ArrayType, 
                     s: ArrayType, 
                     nu: ArrayType, 
                     lam: ArrayType, 
                     L: ArrayType, 
                     b_1: ArrayType, 
                     ) -> ArrayType:
                    
    if parms.loss_type == "mse":
        b = tensor_to_vec(b_1 + v - lam + OPS.G_multop(s - nu, transpose=True))
        bhat = mnp.solve_triangular(L, b, lower=True)
        u = vec_to_tensor(mnp.solve_triangular(L.T, bhat, lower=False), OPS.d, OPS.P_S)
    elif parms.loss_type == "ce":
        raise NotImplementedError("Cross-entropy loss not yet implemented for full ADMM update.")

    return u

"""
Approximates Full ADMM update step using conjugate gradient
"""
def ADMM_cg_update(parms: ADMM_Params, 
                   OPS: FG_Operators, 
                   v: ArrayType, 
                   s: ArrayType, 
                   nu: ArrayType, 
                   lam: ArrayType, 
                   A: ArrayType, 
                   b_1: ArrayType, 
                   ) -> ArrayType:

    """
    @Daniel: Implement CG with preconditioners here
        - use parms.cg_params to specify stuff about preconditioners in admm_utils.ADMM_Params
    """

    assert parms.loss_type == "mse", "Conjugate Gradient can only be used for loss_type=\"mse\"."


    raise NotImplementedError("Conjugate Gradient sovle for ADMM full step still to be impemented.")

    return u