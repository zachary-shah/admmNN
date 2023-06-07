"""
Test performance of all ADMM varients on MNIST with any combination of compute backends 

Script generates 2 plots:
- Train / Val loss / accuracy per ADMM iteration, averaged over n trials
- Solve time, broken down into main ADMM stages (precomputations, u updates, etc)
"""

import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from relu_solver import CReLU_MLP
from utils.relu_utils import binary_classifcation_accuracy
from utils.load_data import load_mnist

"""################ PARAMETERS FOR DATA GENERATION ###################"""
# ----------- Data parameters ------------
dataset_path = "baADMM/datasets/mnist.pkl.gz"
n_mnist = 100 # fixed number of training examples to use
downsample = True # downsample data dim to 100 if True
standardize_data = False # standardize X to zero-mean unit-variance before optimizing

# ----------- Experiment Parameters ------------
P_S =  20 # number of sampled hyperplanes
# backends = ["numpy", "torch", "jax"] # uncomment to run all 3 backend types
backends = ["torch"] # provide a list of backends to try (or just one in list format)
loss_type = 'mse' #either 'mse' or 'ce'
ntrials = 5 # number of trials for each experiment (to average over)
max_iter = 10 # max number of outer optimization iterations
accuracy_func = binary_classifcation_accuracy
bias = True # add bias term to weights
verbose_training = True
seed = None

# ----------- Decide which optimizer methods to generate (at least one below must be "True") -----------
admm_runner = True # to run vanilla admm on selected backends
rbcd_runner = True # to run RBCD on selected backends 
cg_runner = True # to run ADMM with standard conjugate gradient on selected backends
pcg_runner = True # to run ADMM with diagonal (jacobi) preconditioned conjugate gradient on selected backends
nyspcg_runner = True # to run ADMM with nystrom preconditioned conjugate gradient on selected backends

"""#################################################################"""


"""########################### ADMM  ##############################"""
# # ----------- ADMM solver params ------------
admm_base_config = dict(
    optimizer_mode= "ADMM",
    loss_type = loss_type,
    standardize_data = standardize_data,
    P_S = P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias,
    seed=seed,
)

admm_optimizer_configs = [admm_base_config.copy() for k in range(len(backends))]
for (i, backend) in enumerate(backends):
    admm_optimizer_configs[i].update(datatype_backend = backend)
admm_optimizer_labs = [f"ADMM-{backend} (cpu)" for backend in backends]
"""#################################################################"""

"""########################### ADMM-CG ##############################"""
# # ----------- ADMM solver params ------------
admm_cg_base_config = dict(
    optimizer_mode= "ADMM",
    loss_type = loss_type,
    standardize_data = standardize_data,
    P_S = P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias, 
    seed=seed,
    admm_solve_type='cg',
    cg_max_iters=10,
    cg_eps=1e-6,
)

admm_cg_optimizer_configs = [admm_cg_base_config.copy() for k in range(len(backends))]
for (i, backend) in enumerate(backends):
    admm_cg_optimizer_configs[i].update(datatype_backend = backend)
admm_cg_optimizer_labs = [f"ADMM-CG-{backend} (cpu)" for backend in backends]
"""#################################################################"""

"""########################### ADMM-PCG ##############################"""
# # ----------- ADMM solver params ------------
admm_pcg_base_config = dict(
    optimizer_mode= "ADMM",
    loss_type = loss_type,
    standardize_data = standardize_data,
    P_S = P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias, 
    seed=seed,
    admm_solve_type='cg',
    cg_max_iters=10,
    cg_eps=1e-6,
    cg_preconditioner='jacobi'
)

admm_pcg_optimizer_configs = [admm_pcg_base_config.copy() for k in range(len(backends))]
for (i, backend) in enumerate(backends):
    admm_pcg_optimizer_configs[i].update(datatype_backend = backend)
admm_pcg_optimizer_labs = [f"ADMM-Jacobi-PCG-{backend} (cpu)" for backend in backends]
"""#################################################################"""

"""########################### nysADMM ##############################"""
# # ----------- ADMM solver params ------------
nyspcg_base_config = dict(
    optimizer_mode= "ADMM",
    loss_type = loss_type,
    standardize_data = standardize_data,
    P_S = P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias, 
    seed=seed,
    admm_solve_type='cg',
    cg_max_iters=10,
    cg_eps=1e-6,
    cg_preconditioner='nystrom'
)

nyspcg_optimizer_configs = [nyspcg_base_config.copy() for k in range(len(backends))]
for (i, backend) in enumerate(backends):
    nyspcg_optimizer_configs[i].update(datatype_backend = backend)
nyspcg_optimizer_labs = [f"nysADMM-{backend} (cpu)" for backend in backends]
"""#################################################################"""


"""########################### ADMM-RBCD ##############################"""
# # ----------- ADMM solver params ------------
rbcd_base_config = dict(
    optimizer_mode= "ADMM-RBCD",
    loss_type = loss_type,
    standardize_data = standardize_data,
    P_S = P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = bias,
    seed=seed,
)

rbcd_optimizer_configs = [rbcd_base_config.copy() for k in range(len(backends))]
for (i, backend) in enumerate(backends):
    rbcd_optimizer_configs[i].update(datatype_backend = backend)
rbcd_optimizer_labs = [f"ADMM-RBCD-{backend} (cpu)" for backend in backends]
"""#################################################################"""

# ------------ Combine all optimizers desired ----------
optimizer_configs, optimizer_labs = [], []
assert admm_runner or pcg_runner or cg_runner or rbcd_runner or nyspcg_runner, "Must select at least one optimizer runner"
if admm_runner:
    optimizer_configs += admm_optimizer_configs
    optimizer_labs += admm_optimizer_labs
if cg_runner:
    optimizer_configs += admm_cg_optimizer_configs
    optimizer_labs += admm_cg_optimizer_labs
if pcg_runner:
    optimizer_configs += admm_pcg_optimizer_configs
    optimizer_labs += admm_pcg_optimizer_labs
if nyspcg_runner:
    optimizer_configs += nyspcg_optimizer_configs
    optimizer_labs += nyspcg_optimizer_labs
if rbcd_runner:
    optimizer_configs += rbcd_optimizer_configs
    optimizer_labs += rbcd_optimizer_labs


# ------------ PLOT RESULT FOR SOLVER ----------
def plot_metric(metric, 
                x_vals, 
                optimizer_labs, 
                title_str = "", 
                ylabel_str = "", 
                xlabel_str = "",
                use_legend = False):

    metric_mean = np.mean(metric, axis=1)
    metric_std = np.std(metric, axis=1)
    metric_max = np.max(metric, axis=1)
    metric_min = np.min(metric, axis=1)

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
X_train, y_train, X_test, y_test = load_mnist(dataset_rel_path=dataset_path, n=n_mnist, downsample=downsample)

nopt = len(optimizer_configs)
train_loss = np.zeros((nopt, ntrials, max_iter)) * np.nan
train_acc = np.zeros((nopt, ntrials, max_iter)) * np.nan
test_loss = np.zeros((nopt, ntrials, max_iter)) * np.nan
test_acc = np.zeros((nopt, ntrials, max_iter)) * np.nan

solve_time_labels = ["precomps", "u update", "v update", "s update", "dual update"]
solve_time = np.zeros((nopt, ntrials, len(solve_time_labels))) * np.nan

# run for remaining valid optimizers (because cvxpy fails for high data dimension, so don't continue with cvxpy after it fails once)
for (k, optimizer_config) in enumerate(optimizer_configs):
    for t in range(ntrials):
        print(f"{optimizer_labs[k]} trial {t+1}/{ntrials}")
        
        solver = CReLU_MLP(X_train, y_train, **optimizer_config)

        metrics = solver.optimize(max_iter=max_iter, verbose=verbose_training, X_val=X_test, y_val=y_test)

        train_loss[k, t] = metrics["train_loss"]
        train_acc[k, t] = metrics["train_acc"]
        test_loss[k, t] = metrics["val_loss"]
        test_acc[k, t] = metrics["val_acc"]

        # collect solve_time
        solve_time[k, t] = np.array([item for item in metrics["solve_time_breakdown"].values()])

print(f"Done! Generating plots...")

# ------------ plots over solver iteration ----------------
x_vals = np.arange(max_iter)+1

plt.figure()
plt.subplot(221)
plot_metric(train_loss, x_vals, optimizer_labs, 
            title_str = '(a) Train Loss')

plt.subplot(222)
plot_metric(train_acc, x_vals, optimizer_labs, 
            title_str = '(b) Train Accuracy')

plt.subplot(223)
plot_metric(test_loss, x_vals, optimizer_labs, 
            title_str = '(c) Test Loss',
            xlabel_str = "ADMM iteration")

plt.subplot(224)
plot_metric(test_acc, x_vals, optimizer_labs, 
            title_str = '(d) Test Accuracy',
            xlabel_str = "ADMM iteration",
            use_legend=True)
plt.suptitle(f"Figure 1. Performance vs. iteration for loss type: {loss_type}")


# ------------ plots of solve times ----------------

# Create a figure and axis object
fig, ax = plt.subplots()
bar_width = 0.2
x_pos = np.arange(len(solve_time_labels))

solve_times_mean = np.mean(solve_time, axis=1)
for i, lab in enumerate(optimizer_labs):
   pos = x_pos + (i * bar_width)
   ax.bar(pos, solve_times_mean[i], width=bar_width, label=lab)

# Set the x-axis labels and tick positions
ax.set_xticks(x_pos + ((len(optimizer_labs) - 1) / 2) * bar_width)
ax.set_xticklabels(solve_time_labels)

plt.title("Mean solve time breakdown per backend")
plt.xlabel("Computation piece") 
plt.ylabel("Time (s)")
plt.legend()
plt.show()