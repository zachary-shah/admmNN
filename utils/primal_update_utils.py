"""
Main implementations for u updates during ADMM / ADMM-RBCD
"""

from math import ceil
from typing import Tuple

from utils.typing_utils import ArrayType, EvalFunction
from utils.admm_utils import FG_Operators, ADMM_Params, vec_to_tensor, tensor_to_vec
import utils.math_utils as mnp

def RBCD_update(parms: ADMM_Params, 
                OPS: FG_Operators, 
                y: ArrayType, 
                u: ArrayType, 
                v: ArrayType, 
                s: ArrayType, 
                nu: ArrayType, 
                lam: ArrayType, 
                GiTGi: ArrayType, 
                loss_func: EvalFunction,
                verbose: bool = False) -> Tuple[ADMM_Params, ArrayType]:
    """
    Performs RBCD updates step for ADMM-RBCD descent
    """

    if verbose: print(f"  Beginning RBCD update...\n\tloss = {parms.loss_type}")

    u_out = mnp.zeros_like(u)

    # initial estimate
    y_hat = OPS.F_multop(u)

    # precomputation 
    stil = OPS.G_multop(s - nu, transpose=True)

    # unpack variable pairs
    u, z = u[0], u[1]
    v, w = v[0], v[1]
    s, t = s[0], s[1]
    lam1, lam2 = lam[0], lam[1]
    nu1, nu2 = nu[0], nu[1]
    stil, ttil = stil[0], stil[1]

    dcosts = mnp.ones(ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize)), backend_type=parms.datatype_backend) * 1e8
    ptr, k = 0, 0  # k is current count of iterations
    
    max_iter = 1000 #temporary max iteration for debugging; TODO: remove or parameterize
    
    while mnp.mean(dcosts) > parms.RBCD_thresh and k < max_iter:
        k += 1
        i = mnp.random_choice(parms.P_S, size=parms.RBCD_blocksize, replace=False, backend_type=parms.datatype_backend)

        # Calculate training loss via u and z (without regularization) and get gradients
        loss1 = loss_func(y_hat, y)

        if parms.loss_type == 'mse':
            grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (y_hat - y)[:,None])
        elif parms.loss_type == 'ce':
            grad1 = OPS.X.T @ (OPS.d_diags[:, i] * (-2 * y + 2 / (1 + mnp.exp(-2 * y_hat)))[:,None])

        gradu = grad1 + parms.rho * (u[:, i] - v[:, i] + lam1[:, i] + GiTGi @ u[:, i] - stil[:, i])
        gradz = -grad1 + parms.rho * (z[:, i] - w[:, i] + lam2[:, i] + GiTGi @ z[:, i] - ttil[:, i])

        # ----------- Determine the step size using line search -----------------
        alpha = parms.alpha0
        while True:  # Emulate a do-while loop
            du = -alpha * gradu
            dz = -alpha * gradz

            # Current prediction (via convex formulation)
            yhat_new = y_hat + mnp.sum(OPS.d_diags[:, i] * (OPS.X @ (du - dz)), axis=1)

            dloss = loss_func(yhat_new, y) - loss1

            ddist1 = mnp.sum((u[:, i] + du - v[:, i] + lam1[:, i]) ** 2) + \
                     mnp.sum((z[:, i] + dz - w[:, i] + lam2[:, i]) ** 2) - \
                     mnp.sum((u[:, i] - v[:, i] + lam1[:, i]) ** 2) + \
                     mnp.sum((z[:, i] - w[:, i] + lam2[:, i]) ** 2)
            ddist2 = mnp.sum((OPS.e_diags[:, i] * (OPS.X @ (u[:, i] + du)) - s[:, i] + nu1[:, i]) ** 2) + \
                     mnp.sum((OPS.e_diags[:, i] * (OPS.X @ (z[:, i] + dz)) - t[:, i] + nu2[:, i]) ** 2) - \
                     mnp.sum((OPS.e_diags[:, i] * (OPS.X @ u[:, i]) - s[:, i] + nu1[:, i]) ** 2) - \
                     mnp.sum((OPS.e_diags[:, i] * (OPS.X @ z[:, i]) - t[:, i] + nu2[:, i]) ** 2)
            dcost = dloss + (ddist1 + ddist2) * parms.rho / 2

            # Armijo's rule
            if alpha <= 1e-8 or dcost <= -1e-3 * mnp.sqrt(mnp.sum((du ** 2)) + mnp.sum((dz ** 2))):
                break
            alpha /= 2.5

            # Decaying basic step size
            parms.alpha0 = mnp.maximum(1e-10, parms.alpha0 / 1.5)
            
        y_hat = yhat_new 

        # Update u, z, and objective
        if parms.datatype_backend == "jax":
            u = u.at[:, i].add(du)
            z = z.at[:, i].add(dz)
            dcosts = dcosts.at[ptr].add(-dcost)
        else:
            u[:, i] += du
            z[:, i] += dz
            dcosts[ptr] = -dcost

        # Update circular buffer
        ptr = (ptr + 1) % ceil(parms.base_buffer_size * mnp.sqrt(parms.P_S / parms.RBCD_blocksize))
        parms.alpha0 *= 1.05
        if verbose and k % 20 == 0:
            print('\tIteration', k, ', alpha:', alpha, ', delta:', mnp.mean(dcosts))

    # concatenate
    if parms.datatype_backend == "jax":
        u_out = u_out.at[0].set(u)
        u_out = u_out.at[1].set(z)
    else:
        u_out[0] = u
        u_out[1] = z

    return parms, u_out
