"""
Test performance of optimal convex (C-ReLU) vs optimal non-convex transformed weights (NC-ReLU)
No longer works since use of C-ReLU weights has been removed.
"""

import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from datetime import datetime

from relu_solver import Approximate_2_Layer_ReLU
from relu_utils import squared_loss, binary_classifcation_accuracy
from optimizers import admm_optimizer, approx_admm_optimizer, cvxpy_optimizer
from torch.utils.data import DataLoader, TensorDataset

from ADMM_torch import ADMMTrainer
from postprocess import evaluate
from load_data import load_mnist

"""################ PARAMETERS FOR DATA GENERATION ###################"""

# ----------- Data parameters ------------
downsample = True # downsample data dim to 100 if True
standardize_data = False # standardize X to zero-mean unit-variance before optimizing 

# ----------- Experiment Parameters ------------
# TODO: implement ce loss in ADMM
# for RBCD: choose loss of 'ce' or 'mse'
loss_type = 'mse'
ntrials = 5 # number of trials for each experiment (to average over)

# ----------- ADMM solver params ------------
rho  = 0.01 # ADMM param
step = 0.01 # ADMM param
beta = 0.001 # ADMM param
bias = True # add bias term to weights
max_iter = 10 # max iterations for ADMM

# ----------- Choice of optimizers to run ------------
# optimizers = [admm_optimizer, cvxpy_optimizer] # list of functions for optimizers to run
# optimizer_labs = ["ADMM", "CVXPY"] # string labels for each optimizer

optimizers = [admm_optimizer] # list of functions for optimizers to run
optimizer_labs = ["ADMM"] # string labels for each optimizer

# ----------- Figure 1: data varied with P_S params ------------
PS_vals = np.linspace(4,20,4).astype(int) # the P_S values to test against
n_for_varied_P_S = 1000 # fixed number of training examples to use

"""#################################################################"""

# ------------ Train ADMM, get transformed weights too ----------
def run_admm(X_train, y_train, X_test, y_test, params, optimizer, max_iter):
    # using approximate admm solver
    solver = Approximate_2_Layer_ReLU(**params, optimizer=optimizer)

    solver.optimize(X_train, y_train, max_iter=max_iter, verbose=True)

    y_hat_train = solver.predict(X_train)
    y_hat_test = solver.predict(X_test)
    train_loss = squared_loss(y_hat_train, y_train)
    train_acc = binary_classifcation_accuracy(y_hat_train, y_train)
    test_acc = binary_classifcation_accuracy(y_hat_test, y_test)

    return train_loss, train_acc, test_acc

# ------------ PLOT RESULT FOR SOLVER ----------
def plot_metric(metric, 
                x_vals, 
                optimizer_labs, 
                title_str = "", 
                ylabel_str = "", 
                xlabel_str = "",
                use_legend = False):

    metric_mean = np.mean(metric, axis=2)
    metric_std = np.std(metric, axis=2)
    metric_max = np.max(metric, axis=2)
    metric_min = np.min(metric, axis=2)

    full_labs = optimizer_labs

    for indx, lab in enumerate(full_labs):
        upper_error = np.minimum(metric_max[indx], metric_mean[indx] + metric_std[indx])
        lower_error = np.maximum(metric_min[indx], metric_mean[indx] - metric_std[indx])

        plt.plot(x_vals, metric_mean[indx], '--o', label=lab)
        plt.fill_between(x_vals, lower_error, upper_error, alpha=0.2)

    plt.title(title_str)
    plt.xlabel(xlabel_str) 
    plt.ylabel(ylabel_str)
    if use_legend: plt.legend()


# ------------ Load Data ------------
print(f'Loading data...')
# Load mnist and select only digts 2 and 8, but only get 1000 samples
X_train, y_train, X_test, y_test = load_mnist(n=-1, downsample=downsample)
yy_train, yy_test = ((y_train+1)//2).astype(int), ((y_test+1)//2).astype(int)

# Show dims
print(f'Data loaded. Full data dimensions: ')
print(f'  X_train shape = {X_train.shape}')
print(f'  X_test  shape = {X_test.shape}')
print(f'  y_train shape = {y_train.shape}')
print(f'  y_test  shape = {y_test.shape}')
print(f"  Proportion of 8s in train data: {np.sum(y_train == 1)/len(y_train)}")
print(f"  Proportion of 8s in test data: {np.sum(y_test == 1)/len(y_test)}")

# ------------ Figure 1: Accuracy and time vs. P_S ------------
# get subset of train data
X_tr = X_train[:n_for_varied_P_S, :]
y_tr = y_train[:n_for_varied_P_S]
yy_tr = yy_train[:n_for_varied_P_S]

print(f'Generating Figure 1 data...')
print(f"  Proportion of 8s in train data: {np.sum(y_tr == 1)/len(y_tr)}")
print(f"  Proportion of 8s in test data: {np.sum(y_test == 1)/len(y_test)}")

# values of n to try
nopt = len(optimizers)
train_loss = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
train_acc = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
test_acc = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
solve_time = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan

for (i,P_S) in enumerate(PS_vals):

    print(f"TRIALS FOR P_S={P_S} (size {i+1}/{len(PS_vals)})")

    params = dict(m=P_S,
             P_S=P_S,
             rho=rho,
             step=step,
             beta=beta,
             bias=bias,
             standardize_data=standardize_data,
             acc_func=binary_classifcation_accuracy,
    )

    for t in range(ntrials):
        # run for remaining valid optimizers (because cvxpy fails for high data dimension, so don't continue with cvxpy after it fails once)
        for (k, optimizer) in enumerate(optimizers):
            print(f"  {optimizer_labs[k]} trial {t+1}/{ntrials}")
            tl, ta, tea = run_admm(X_tr, y_tr,X_test,y_test,params,optimizer,max_iter)
            train_loss[k, i, t] = tl
            train_acc[k,i, t] = ta
            test_acc[k,i,t] = tea

print(f"Done! Generating plots...")

# ------------ plots for varied P_S ----------------
plt.subplot(131)
plot_metric(train_loss, PS_vals, optimizer_labs, 
            title_str = '(a) Train Loss vs. P_S',
            xlabel_str = "Number of hidden layers P_S")

plt.subplot(132)
plot_metric(train_acc, PS_vals, optimizer_labs, 
            title_str = '(c) Train Accuracy vs. P_S',
            xlabel_str = "Number of hidden layers P_S")

plt.subplot(133)
plot_metric(test_acc, PS_vals, optimizer_labs, 
            title_str = '(d) Test Accuracy vs. P_S',
            xlabel_str = "Number of hidden layers P_S",
            use_legend=True)

plt.suptitle(f"Figure 1. Performance vs. P_S for loss type: {loss_type}")
plt.show()