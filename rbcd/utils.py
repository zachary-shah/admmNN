# Math
import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import randn
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag, solve_triangular, solve, cholesky
from scipy.optimize import minimize
from math import ceil


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return (z>=0).astype(int)
    # return ((z>=0).astype(int)*1+0)


def proxl2(z, beta, gamma, use_torch=False):
    norm_func = torch.linalg.norm if use_torch else np.linalg.norm
    relu_func = F.relu if use_torch else relu
    zeros_like_func = torch.zeros_like if use_torch else np.zeros_like

    if len(list(z.shape)) == 1:  # One-dimensional
        if norm_func(z) == 0:
            return z
        else:
            return relu_func(1 - beta * gamma / norm_func(z)) * z
    elif len(list(z.shape)) == 2:  # Two-dimensional
        norms = norm_func(z, axis=0)
        mask = norms > 0
        res = zeros_like_func(z)
        res[:, mask] = relu_func(1 - beta * gamma / norms[mask]) * z[:, mask]
        return res
    else:
        raise('Wrong dimensions')


# Generate D Matrices
def generate_D(X, P, v=-1, w=-1, verbose=False):
    (n, d) = X.shape
    X = X.astype(np.float32)
    if w == -1 and v == -1:
        v = randn(d, P).astype(np.float32)
        dmat, ind = np.unique(relu_prime(X @ v), axis=1, return_index=True)
        v = v[:, ind]
        if verbose: print((2 * dmat-1) * (X @ v) >= 0)
    else:
        P = v.shape[1]
        dmat1 = relu_prime(X @ v)
        dmat2 = relu_prime(X @ w)
        dmat = np.concatenate([dmat1, dmat2], axis=1)
        temp, ind = np.unique(dmat, axis=1, return_index=True)
        ind1 = ind[ind < P]
        ind2 = ind[ind >= P] - P
        dmat = dmat[:, np.concatenate([ind1, ind2+P])]
        wnew = w[:, ind2]
        v, w = v[:, ind1], w[:, ind1]
        w[:, ind2] = np.zeros([d, ind2.size])
        w = np.concatenate([w, wnew], axis=1)
        v = np.concatenate([v, np.zeros([d, ind2.size])], axis=1)
        if verbose: print((2*dmat-1) * (X @ v) >= 0)
        if verbose: print((2*dmat-1) * (X @ w) >= 0)
    return dmat, n, d, dmat.shape[1], v, w
