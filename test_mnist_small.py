"""
Test performance of just ADMM vs ADMM-RBCD on MNIST
"""

import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from relu_solver import CReLU_MLP
from utils.relu_utils import squared_loss, binary_classifcation_accuracy
from utils.load_data import load_mnist

"""################ PARAMETERS FOR DATA GENERATION ###################"""

# ----------- Data parameters ------------
dataset_path = "baADMM/datasets/mnist.pkl.gz"
downsample = True # downsample data dim to 100 if True
standardize_data = False # standardize X to zero-mean unit-variance before optimizing
bias = True # add bias term to weights
max_iter = 35
accuracy_func = binary_classifcation_accuracy
verbose_training = True
seed = 527364

# ----------- Experiment Parameters ------------
loss_type = 'mse' #either 'mse' or 'ce'
ntrials = 1 # number of trials for each experiment (to average over)

# ----------- ADMM solver params ------------
admm_config = dict(
    optimizer_mode= "ADMM",
    loss_type = loss_type,
    standardize_data = standardize_data,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias,
    rho=0.01,
    step=0.01,
    beta=0.001, 
    seed=seed,
    admm_cg_solve=True,
)

# ----------- ADMM-RBCD solver params ------------
rbcd_config = dict(
    optimizer_mode= "ADMM-RBCD",
    loss_type = 'ce',
    standardize_data = standardize_data,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias=bias,
    rho=0.02,
    beta=0.0001, 
    alpha0=3e-6,
    RBCD_blocksize=3,
    RBCD_thresh=.7,
    gamma_ratio=0.2,
    seed=seed,
)

optimizer_configs = [admm_config, rbcd_config]
optimizer_labs = ["ADMM", "ADMM-RBCD"] # string labels for each optimizer

# ----------- Figure 1: data varied with P_S params ------------
PS_vals = [4, 24] # np.linspace(4,20,4).astype(int) # the P_S values to test against
n_for_varied_P_S = 1000 # fixed number of training examples to use


"""#################################################################"""

# ------------ Train ADMM, get transformed weights too ----------
def run_admm(X_train, y_train, X_test, y_test, params, max_iter):
    # using approximate admm solver
    solver = CReLU_MLP(X_train, y_train, **params)

    solver.optimize(max_iter=max_iter, verbose=verbose_training)

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

    for indx, lab in enumerate(optimizer_labs):
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
X_train, y_train, X_test, y_test = load_mnist(dataset_rel_path=dataset_path, n=-1, downsample=downsample)
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
nopt = len(optimizer_configs)
train_loss = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
train_acc = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
test_acc = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan
solve_time = np.zeros((nopt, len(PS_vals),ntrials)) * np.nan

for (i,P_S) in enumerate(PS_vals):

    print(f"TRIALS FOR P_S={P_S} (size {i+1}/{len(PS_vals)})")

    for t in range(ntrials):
        # run for remaining valid optimizers (because cvxpy fails for high data dimension, so don't continue with cvxpy after it fails once)
        for (k, optimizer_config) in enumerate(optimizer_configs):

            optimizer_config["P_S"] = P_S

            print(f"  {optimizer_labs[k]} trial {t+1}/{ntrials}")

            # [0, 1] labels for ADMM-RBCD only
            if optimizer_config == "ADMM-RBCD":
                tl, ta, tea = run_admm(X_tr, yy_tr, X_test, yy_test, optimizer_config, max_iter)
            else:
                tl, ta, tea = run_admm(X_tr, y_tr, X_test, y_test, optimizer_config, max_iter)

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