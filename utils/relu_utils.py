"""
Losses and Classifcation Functions
"""

from utils.typing_utils import ArrayType

import utils.math_utils as mnp

"""
Calculates the squared loss 1/2||y_hat - y||_2^2
"""
def squared_loss(y_hat: ArrayType,
                 y: ArrayType) -> float:

    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"

    return 0.5 * mnp.norm(y_hat - y) ** 2

"""
Calculates the squared loss 1/2||y_hat - y||_2^2
"""
def cross_entropy_loss(y_hat: ArrayType,
                       y: ArrayType) -> float:

    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"

    return -2 * mnp.dot(y, y_hat) + mnp.sum(mnp.log(mnp.exp(2 * y_hat) + 1))

"""
Calculates accuracy for classification problem
"""
def classifcation_accuracy(y_hat: ArrayType,
                           y: ArrayType) -> float:

    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = mnp.round(y_hat)
    return mnp.sum(y_hat == y) / len(y)

"""
Calculates accuracy for binary classification problem
"""
def binary_classifcation_accuracy(y_hat: ArrayType,
                                  y: ArrayType) -> float:

    if len(y_hat.shape) == 2:
        y_hat = mnp.reshape(y_hat, (-1))
    if len(y.shape) == 2:
        y = mnp.reshape(y, (-1))

    assert len(y_hat) == len(y), f"y_hat (n={len(y_hat)}) and y (n={len(y)}) are different lengths"
    
    y_hat = mnp.sign(y_hat)
    return mnp.sum(y_hat == y) / len(y)