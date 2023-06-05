"""
Losses and Classifcation Functions
"""

from typing import Tuple
from utils.typing_utils import ArrayType, EvalFunction, get_backend_type

import utils.math_utils as mnp

def squared_loss(y_hat: ArrayType,
                 y: ArrayType) -> float:
    
    """
    Calculates the squared loss 1/2||y_hat - y||_2^2
    """

    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"

    return 0.5 * mnp.norm(y_hat - y) ** 2

def cross_entropy_loss(y_hat: ArrayType,
                       y: ArrayType) -> float:
    
    """
    Calculates cross entropy loss
    """
    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"

    return -2 * mnp.dot(y, y_hat) + mnp.sum(mnp.log(mnp.exp(2 * y_hat) + 1))

def classifcation_accuracy(y_hat: ArrayType,
                           y: ArrayType) -> float:
    
    """
    Calculates accuracy for classification problem
    """
    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = mnp.round(y_hat)
    return mnp.sum(y_hat == y) / len(y)

def binary_classifcation_accuracy(y_hat: ArrayType,
                                  y: ArrayType) -> float:

    """
    Calculates accuracy for binary classification problem
    """
    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = mnp.sign(y_hat)
    return mnp.sum(y_hat == y) / len(y)

def optimal_weights_transform(v: ArrayType, 
                              w: ArrayType, 
                              P_S: int, 
                              d: int,
                              verbose: bool = False) -> Tuple[ArrayType, ArrayType]:
    """
    Given optimal v^*, w^* of convex problem (Eq (2.1)), derive the optimal weights u^*, alpha^* of the non-convex probllem (Eq (2.1))
    Applies Theorem 1 of Pilanci, Ergen 2020

    Parameters
    ----------
    v : ArrayType
        v weights in convex formulation
    w : ArrayType
        w weights in convex formulation
    P_S : int
        number of hyperplane samples, for data dimension validation
    d: int
        number of features, for data dimension validatation
    verbose: boolean
        true to print weight transform information
   
    Returns
    -------
    (u, alpha) : Tuple[ArrayType, ArrayType]
        the transformed optimal weights
    """

    assert v is not None
    assert w is not None

    # ensure shapes are correct
    if v.shape == (P_S, d): v = v.T
    if w.shape == (P_S, d): w = w.T
    assert v.shape == (d, P_S), f"Expected weight v shape to be ({d},{P_S}), but got {v.shape}"
    assert w.shape == (d, P_S), f"Expected weight w shape to be ({d},{P_S}), but got {w.shape}"

    if verbose: 
        datatype_backend = get_backend_type(v)
        print(f"\nDoing weight transform: ")
        v_shp = v.cpu().numpy().shape if datatype_backend == "torch" else v.shape
        w_shp = w.cpu().numpy().shape if datatype_backend == "torch" else w.shape
        print(f"  starting v shape: {v_shp}")
        print(f"  starting w shape: {w_shp}")
        print(f"  (d, P_S): ({d}, {P_S})")

    alpha1 = mnp.sqrt(mnp.norm(v, 2, axis=0))
    mask1 = alpha1 != 0
    u1 = v[:, mask1] / alpha1[mask1]
    alpha2 = -mnp.sqrt(mnp.norm(w, 2, axis=0))
    mask2 = alpha2 != 0
    u2 = -w[:, mask2] / alpha2[mask2]

    u = mnp.append(u1, u2, axis=1)
    alpha = mnp.append(alpha1[mask1], alpha2[mask2])

    if verbose: 
        u_shp = u.cpu().numpy().shape if datatype_backend == "torch" else u.shape
        a_shp = alpha.cpu().numpy().shape if datatype_backend == "torch" else alpha.shape
        print(f"  transformed u shape: {u_shp}")
        print(f"  transformed alpha shape: {a_shp}")  

    return u, alpha