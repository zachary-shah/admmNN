# PyTorch and Numpy
import torch
import torch.nn.functional as F
import numpy as np

# Helper functions
from utils import relu


def calculate_cost(X, y, u, alpha, beta=1e-4, loss_type='mse', use_torch=False, verbose=True):
    relu_func = F.relu if use_torch else relu
    log_func = torch.log if use_torch else np.log
    exp_func = torch.exp if use_torch else np.exp
    
    yhat = relu_func(X @ u) @ alpha
    if loss_type == 'mse':
        total_cost = ((yhat - y) ** 2).sum() / 2 + beta * (u ** 2).sum() / 2 + beta * (alpha ** 2).sum() / 2
    elif loss_type == 'ce':
        total_cost = (-2 * yhat * y + log_func(exp_func(2 * yhat) + 1)).sum() + \
                      beta / 2 * ((u ** 2).sum() + (alpha ** 2).sum())

    if verbose: print("Total cost: ", total_cost.item() if use_torch else total_cost)
    return total_cost


def evaluate2(X, y, u, alpha, use_torch=False, verbose=True):
    def evaluate2_np(X, y, u, alpha, verbose=True):
        yhat = relu(X @ u) @ alpha
        if y.min() < -1e-3:  # 1, -1 binary classification
            yhat = (yhat > 0).astype(int)
            accuracy = (yhat == (y.astype(int)+1) // 2).sum() / y.size
        else:  # 1, 0 binary classification
            yhat = (yhat > .5).astype(int)
            accuracy = (yhat == y.astype(int)).sum() / y.size
        if verbose: print("Accuracy: ", accuracy)
        return accuracy, yhat

    def evaluate2_torch(X, y, u, alpha, verbose=True):
        yhat = F.relu(X @ u) @ alpha
        if y.min() < -1e-3:  # 1, -1 binary classification
            yhat = (yhat > 0).int()
            accuracy = (yhat == (y.int()+1) // 2).sum() / y.size()[0]
        else:  # 1, 0 binary classification
            yhat = (yhat > .5).int()
            accuracy = (yhat == y.int()).sum() / y.size()[0]
        if verbose: print("Accuracy: ", accuracy.item())
        return accuracy.item(), yhat

    return evaluate2_torch(X, y, u, alpha, verbose) if use_torch \
        else evaluate2_np(X, y, u, alpha, verbose)


def evaluate(X, y, u, alpha, use_torch=False, verbose=True):
    def evaluate_np(X, y, u, alpha, verbose=True):
        yhat = relu(X @ u) @ alpha
        if y.min() < -1e-3:  # 1, -1 binary classification
            yhat = (yhat > 0).astype(int)
            accuracy = (yhat == (y.astype(int)+1) // 2).sum() / y.size
        else:  # 1, 0 binary classification
            yhat = (yhat > 0).astype(int)
            accuracy = (yhat == y.astype(int)).sum() / y.size
        if verbose: print("Accuracy: ", accuracy)
        return accuracy, yhat

    def evaluate_torch(X, y, u, alpha, verbose=True):
        yhat = F.relu(X @ u) @ alpha
        if y.min() < -1e-3:  # 1, -1 binary classification
            yhat = (yhat > 0).int()
            accuracy = (yhat == (y.int() + 1) // 2).sum() / y.size()[0]
        else:  # 1, 0 binary classification
            yhat = (yhat > 0).int()
            accuracy = (yhat == y.int()).sum() / y.size()[0]
        if verbose: print("Accuracy: ", accuracy.item())
        return accuracy.item(), yhat

    return evaluate_torch(X, y, u, alpha, verbose) if use_torch \
        else evaluate_np(X, y, u, alpha, verbose)


def recover_weights(v, w, use_torch=False, verbose=False):  # Recover u, alpha from v, w
    norm_func = torch.linalg.norm if use_torch else np.linalg.norm
    sqrt_func = torch.sqrt if use_torch else np.sqrt
    
    alpha1 = sqrt_func(norm_func(v, 2, axis=0))
    mask1 = alpha1 != 0
    u1 = v[:, mask1] / alpha1[mask1]
    alpha2 = -sqrt_func(norm_func(w, 2, axis=0))
    mask2 = alpha2 != 0
    u2 = -w[:, mask2] / alpha2[mask2]

    u = torch.cat((u1, u2), dim=1) if use_torch else np.append(u1, u2, axis=1)
    alpha = torch.cat((alpha1[mask1], alpha2[mask2])) if use_torch \
        else np.append(alpha1[mask1], alpha2[mask2])

    if verbose: 
        print((u.cpu().numpy(), alpha.cpu().numpy()) if use_torch else (u, alpha))
    return u, alpha
