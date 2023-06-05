"""
Utility functions for preconditioners (nystrom, sketch, etc.)
"""
from typing import Tuple
from utils.typing_utils import ArrayType
from utils.typing_utils import get_backend_type
import utils.math_utils as mnp

def nystrom_sketch(A: ArrayType, 
                   rank: int) -> Tuple[ArrayType, ArrayType]:
    """
    Computes the Nystrom approximation via sketch A.T@(A@Omega) following Tropp at al. 2017

    Parameters
    ----------
    A: ArrayType
        matrix to precondition
    rank : int
        number of top eigenvalues to flatten in preconditioning

    Returns
    ----------
    U : ArrayType
        first preconditioning matrix
    S : ArrayType
        second preconditioning matrix
    """

    backend_type = get_backend_type(A)

    m, n = A.shape

    Omega = mnp.randn((n,rank), backend_type=backend_type) #Generate test matrix
    Omega = mnp.qr(Omega)[0]

    Y = A.T @ (A @ Omega) # Compute sketch

    v = mnp.sqrt(rank) * mnp.spacing(mnp.norm(Y)) #Compute shift according to Martinsson & Tropp 2020
    Yv = Y + v * Omega # Add shift

    Core = Omega.T @ Yv

    try:
        C = mnp.cholesky(Core) #Do Cholesky on Core
    except:
        print("Failed cholesky on core. doing SVD and adding shift instead.")

        eig_vals = mnp.eigh(Core, eigvals_only=True) #If Cholesky fails do SVD and add shift for it to succeed

        v = v + mnp.abs(mnp.min(eig_vals))

        Core = Core + v * mnp.eye(rank, backend_type=backend_type)

        C = mnp.cholesky(Core)

    B = mnp.solve_triangular(C, Yv.T, lower = True, check_finite = False)

    U, S, V = mnp.svd(B.T)

    S = mnp.relu(S**2 - v) # Subtract off shift

    return U, S

def random_sketch(A, rank):

    """
    Computes Random sketch following mert's paper
    """

    raise NotImplementedError("TODO: implement random sketching according to mert's paper")

    return U, S

def hadamard(m, backend_type):
    """
    Computes mxm hadamard matrix
    """

    if m == 2:
        return mnp.array([[1, 1],
                         [1, -1]], backend_type=backend_type)
    else:
        Hm2 = hadamard(m//2, backend_type)
        row2 = mnp.hstack((Hm2, -Hm2))
        row1 = mnp.hstack((Hm2, Hm2))
        return mnp.vstack((row1, row2))