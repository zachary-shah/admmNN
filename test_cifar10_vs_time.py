"""
Test performance of all ADMM varients on CIFAR-10 with any combination of compute args.backends 

Script generates 2 plots:
- Train / Val loss / accuracy vs compute time, averaged over n trials
- Solve time, broken down into main ADMM stages (precomputations, u updates, etc)
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from relu_solver import CReLU_MLP
from utils.relu_utils import classifcation_accuracy, binary_classifcation_accuracy, squared_loss, cross_entropy_loss
from utils.load_data import load_cifar
from datetime import datetime
dtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

"""################ PARAMETERS FOR DATA GENERATION ###################"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADMM methods on CIFAR-10."
    )
    
    parser.add_argument("--dataset_path", type=str, default="baADMM/datasets/cifar-10-batches-py") # path to cifar model
    parser.add_argument("--n_train", type=int, default=1000) # fixed number of training examples to use
    parser.add_argument("--n_test", type=int, default=1000) # fixed number of training examples to use
    parser.add_argument("--ntrials", type=int, default=5) # number of trials
    parser.add_argument("--max_admm_iter", type=int, default=10) # max number of admm iterations
    parser.add_argument("--max_rbcd_iter", type=int, default=100) # max number of rbcd-admm iterations
    parser.add_argument("--max_backprop_iter", type=int, default=1000) # max number of epochs for backprop version
    parser.add_argument("--max_time", type=int, default=120) # max seconds for admm iterations
    parser.set_defaults(downsample=True)
    parser.set_defaults(binary_classes=True)
    parser.add_argument('--no_downsample', dest='downsample', action='store_false', default=True) # downsample data dim to 100 if True
    parser.add_argument('--all_classes', dest='binary_classes', action='store_false') # decide if do 2 classes or 10 classes

    parser.add_argument("--standardize_data", action='store_true', default=False) # standardize data
    parser.add_argument("--P_S", type=int, default=10) # number of sampled hyperplanes
    parser.add_argument("--save_root", type=str, default="figures/cifar-10") # where to save figures to
    parser.add_argument("--backend", type=str, default="torch") # select compute backend
    parser.add_argument("--devices", nargs='+', type=str, default=["cpu", "cuda"]) # devices to use
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--seed", type=int, default=None)

    parser.set_defaults(verbose_training=True)
    parser.add_argument("--silent_training", dest="verbose_training", action="store_false")

    parser.set_defaults(admm_runner=True)
    parser.set_defaults(rbcd_runner=True)
    parser.set_defaults(cg_runner=True)
    parser.set_defaults(pcg_runner=True)
    parser.set_defaults(nyspcg_runner=True)
    parser.set_defaults(backprop_runner=True)
    parser.add_argument("--admm_runner_off", dest="admm_runner", action="store_false")
    parser.add_argument("--rbcd_runner_off", dest="rbcd_runner", action="store_false")
    parser.add_argument("--cg_runner_off", dest="cg_runner", action="store_false")
    parser.add_argument("--pcg_runner_off", dest="pcg_runner", action="store_false")
    parser.add_argument("--nyspcg_runner_off", dest="nyspcg_runner", action="store_false")
    parser.add_argument("--backprop_runner_off", dest="backprop_runner", action="store_false")
    parser.add_argument("--memory_save", action="store_true", default=False)

    args = parser.parse_args()
    return args

args = parse_args()
cg_max_iters = 50
# ----------- Data parameters ------------

# ----------- Decide which optimizer methods to generate (at least one below must be "True") -----------
accuracy_func = binary_classifcation_accuracy if args.binary_classes else classifcation_accuracy
loss_func = cross_entropy_loss if args.loss_type=="ce" else squared_loss

devices = args.devices
print(devices)
admm_runner = args.admm_runner # to run vanilla admm on selected args.backends
rbcd_runner = args.rbcd_runner # to run RBCD on selected args.backends 
cg_runner = args.cg_runner # to run ADMM with standard conjugate gradient on selected args.backends
pcg_runner = args.pcg_runner # to run ADMM with diagonal (jacobi) preconditioned conjugate gradient on selected args.backends
nyspcg_runner = args.nyspcg_runner # to run ADMM with nystrom preconditioned conjugate gradient on selected args.backends
backprop_runner = args.backprop_runner # train a nn with backpropagation for comparison 

"""#################################################################"""

"""########################### CONFIGS  ##############################"""
# # ----------- ADMM solver params ------------
admm_optimizer_config = dict(
    optimizer_mode= "ADMM",
    datatype_backend = args.backend,
    loss_type = args.loss_type,
    standardize_data = args.standardize_data,
    P_S = args.P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = True,
    seed=args.seed,
    memory_save = args.memory_save,
)

# # ----------- ADMM-CG solver params ------------
cg_optimizer_config = dict(
    optimizer_mode= "ADMM",
    datatype_backend = args.backend,
    loss_type = args.loss_type,
    standardize_data = args.standardize_data,
    P_S = args.P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = True,
    seed=args.seed,
    memory_save = args.memory_save,
    admm_solve_type='cg',
    cg_max_iters=cg_max_iters,
    cg_eps=1e-6,
)
# # ----------- ADMM-PCG-Jacobian solver params ------------
pcg_optimizer_config = dict(
    optimizer_mode= "ADMM",
    datatype_backend = args.backend,
    loss_type = args.loss_type,
    standardize_data = args.standardize_data,
    P_S = args.P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = True,
    seed=args.seed,
    memory_save = args.memory_save,
    admm_solve_type='cg',
    cg_max_iters=cg_max_iters,
    cg_eps=1e-6,
    cg_preconditioner='jacobi',
)
# # ----------- ADMM-nysPCG solver params ------------
nys_optimizer_config = dict(
    optimizer_mode= "ADMM",
    datatype_backend = args.backend,
    loss_type = args.loss_type,
    standardize_data = args.standardize_data,
    P_S = args.P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = True,
    seed=args.seed,
    memory_save = args.memory_save,
    admm_solve_type='cg',
    cg_max_iters=cg_max_iters,
    cg_eps=1e-6,
    cg_preconditioner='nystrom',
)
# # ----------- RBCD solver params ------------
rbcd_optimizer_config = dict(
    optimizer_mode= "ADMM-RBCD",
    datatype_backend = args.backend,
    loss_type = args.loss_type,
    standardize_data = args.standardize_data,
    P_S = args.P_S,
    acc_func = accuracy_func,
    verbose_initialization=True,
    bias = True,
    seed=args.seed,
    memory_save = args.memory_save,
)
# # ----------- Backprop solver params ------------
backprop_config = dict(
    optimizer_mode= "Backprop",
    loss_type = args.loss_type,
    loss_func = loss_func,
    acc_func = accuracy_func,
    m = args.P_S,
    lr = 1e-5,
    batch_size = 8,
    epochs = args.max_backprop_iter, 
    seed=args.seed,
)
"""#################################################################"""

# ------------ Combine all optimizers desired ----------
optimizer_configs, optimizer_labs = [], []
device_strs = list(map(lambda x: x.replace('cuda', 'gpu'), devices))
assert admm_runner or pcg_runner or cg_runner or rbcd_runner or nyspcg_runner or backprop_runner, "Must select at least one optimizer runner"
if admm_runner:
    optimizer_configs.append(admm_optimizer_config)
    for device in device_strs: optimizer_labs.append(f"ADMM-({device})")
if cg_runner:
    optimizer_configs.append(cg_optimizer_config)
    for device in device_strs: optimizer_labs.append(f"ADMM-CG-({device})")
if pcg_runner:
    optimizer_configs.append(pcg_optimizer_config)
    for device in device_strs: optimizer_labs.append(f"ADMM-diagPCG-({device})")
if nyspcg_runner:
    optimizer_configs.append(nys_optimizer_config)
    for device in device_strs: optimizer_labs.append(f"ADMM-nysPCG-({device})")
if rbcd_runner:
    optimizer_configs.append(rbcd_optimizer_config)
    for device in device_strs: optimizer_labs.append(f"ADMM-RBCD-({device})")
if backprop_runner:
    optimizer_configs.append(backprop_config)
    for device in device_strs: optimizer_labs.append(f"NCVX-Adam-({device})")

# ------------ Train MLP with adam optimizer via backpropagation ------------
def run_nn_mlp(X_train, y_train, X_test, y_test, config):

    loss_type = config["loss_type"]
    acc_func = config["acc_func"]
    myloss_func = config["loss_func"]
    m = config["m"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = config["device"]

    # simple 2 layer relu network
    MLP = nn.Sequential(nn.Linear(X_train.shape[1], m),
                        nn.ReLU(),
                        nn.Linear(m, 1)).to(device)

    # Train/Test data to torch
    X_train_torch = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_torch = torch.as_tensor(y_train[:, None], dtype=torch.float32, device=device)
    X_test_torch = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    trainset = TensorDataset(X_train_torch, y_train_torch)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss
    optim = torch.optim.Adam(MLP.parameters(), lr=lr)

    if loss_type=="mse":
        loss_func = nn.MSELoss()
    elif loss_type=="ce":
        loss_func = nn.BCELoss()
    else:
        raise NotImplementedError("Loss must be either mse or ce.")

    # Train
    start = time.perf_counter()
    train_loss, train_acc, test_loss, test_acc, solve_timepoints = [], [], [], [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}', end='\r', flush=True)
        for sample, label in iter(trainloader):

            # Move to device
            sample = sample.to(device)
            label = label.to(device)

            # predict 
            pred = MLP(sample)

            # Back prop
            loss = loss_func(pred, label)
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Validataion prediction
        if epoch % 10 == 0:
            with torch.no_grad():
                y_hat_train = MLP(X_train_torch).to('cpu').numpy()
                y_hat_test = MLP(X_test_torch).to('cpu').numpy()

                train_loss.append(myloss_func(y_hat_train, y_train))
                train_acc.append(acc_func(y_hat_train, y_train))
                test_loss.append(myloss_func(y_hat_test, y_test))
                test_acc.append(acc_func(y_hat_test, y_test))
                solve_timepoints.append(time.perf_counter() - start)

        if time.perf_counter() - start > args.max_time: break

    solve_time = time.perf_counter() - start

    return train_loss, test_loss, train_acc, test_acc, solve_time, solve_timepoints

# ------------ PLOT RESULT FOR SOLVER ----------
def plot_metric(metric, 
                timepoints, 
                optimizer_labs, 
                title_str = "", 
                xlabel_str = "",
                ylabel_str = "", 
                x_logscale = False,
                y_logscale = False,
                use_legend = False):

    for indx, lab in enumerate(optimizer_labs):

        # find longest iteration count for a trial
        max_len = max([len(arr) for arr in metric[indx]])

        # pad each to max number of iterations in case iterations terminated early
        def nanpad(arr): return np.pad(arr, (0, max_len - len(arr)), mode='constant',constant_values=(np.nan,np.nan))
        metric_padded, timepoints_padded = [], []
        for trial_indx in range(len(metric[indx])):
            metric_padded.append(nanpad(metric[indx, trial_indx]))
            timepoints_padded.append(nanpad(timepoints[indx, trial_indx]))

        # turn into 2d arrays
        metric_padded = np.stack(metric_padded)
        timepoints_padded = np.stack(timepoints_padded)

        metric_mean = np.nanmean(metric_padded, axis=0)
        metric_std = np.nanstd(metric_padded, axis=0)
        metric_max = np.nanmax(metric_padded, axis=0)
        metric_min = np.nanmin(metric_padded, axis=0)
        timepoints_padded = np.nanmean(timepoints_padded, axis=0)

        upper_error = np.minimum(metric_max, metric_mean + metric_std)
        lower_error = np.maximum(metric_min, metric_mean - metric_std)

        plt.plot(timepoints_padded, metric_mean, '-', label=lab)
        plt.fill_between(timepoints_padded, lower_error, upper_error, alpha=0.2)
        
        if x_logscale: plt.xscale('log')
        if y_logscale: plt.yscale('log')


    plt.title(title_str)
    plt.xlabel(xlabel_str) 
    plt.ylabel(ylabel_str)
    if use_legend: plt.legend(facecolor='white',frameon = True)

# ------------ Load Data ------------
print(f'Loading data...')
# Load mnist and select only digts 2 and 8, but only get 1000 samples
os.makedirs(args.save_root, exist_ok=True)
X_train, y_train, X_test, y_test = load_cifar(dataset_rel_path=args.dataset_path, n=args.n_train, downsample=args.downsample, binary_classes=args.binary_classes)
X_test, y_test = X_test[:args.n_test], y_test[:args.n_test]

print(f'  X_train shape = {X_train.shape}')
print(f'  X_test  shape = {X_test.shape}')
print(f'  y_train shape = {y_train.shape}')
print(f'  y_test  shape = {y_test.shape}')

nopt = len(optimizer_configs) * len(devices)
train_loss = np.empty((nopt, args.ntrials), dtype=object)
train_acc = np.empty((nopt, args.ntrials), dtype=object)
test_loss = np.empty((nopt, args.ntrials), dtype=object)
test_acc = np.empty((nopt, args.ntrials), dtype=object)
solve_timepoints = np.empty((nopt, args.ntrials), dtype=object)

solve_time_labels = ["total_time", "precomps", "u update", "v update", "s update", "dual update"]
solve_time = np.zeros((nopt, args.ntrials, len(solve_time_labels))) * np.nan

# run for each optimizer and device
for (k, optimizer_config) in enumerate(optimizer_configs):
    for t in range(args.ntrials):
        for (d_no, dev) in enumerate(devices):

            if args.backend == "torch": torch.cuda.empty_cache()

            print(f"{optimizer_labs[k]} trial {t+1}/{args.ntrials}")

            optimizer_config.update(device=dev)

            if optimizer_labs[len(devices) * k + d_no].startswith("NCVX"):
                rl, el, ra, ea, st, stps = run_nn_mlp(X_train, y_train, X_test, y_test, optimizer_config)
                train_loss[len(devices) * k + d_no, t] = rl
                train_acc[len(devices) * k + d_no, t] = ra
                test_loss[len(devices) * k + d_no, t] = el
                test_acc[len(devices) * k + d_no, t] = ea
                solve_timepoints[len(devices) * k + d_no, t] = stps
                solve_time[len(devices) * k + d_no, t, 0] = st # only put solve time into u update
            else:
                solver = CReLU_MLP(X_train, y_train, **optimizer_config)
                max_iter = args.max_admm_iter if optimizer_config["optimizer_mode"] == "ADMM" else args.max_rbcd_iter
                metrics = solver.optimize(max_iter=max_iter, max_time=args.max_time, verbose=args.verbose_training, X_val=X_test, y_val=y_test)
                train_loss[len(devices) * k + d_no, t] = metrics["train_loss"]
                train_acc[len(devices) * k + d_no, t] = metrics["train_acc"]
                test_loss[len(devices) * k + d_no, t] = metrics["val_loss"]
                test_acc[len(devices) * k + d_no, t] = metrics["val_acc"]
                solve_timepoints[len(devices) * k + d_no, t] = metrics["iteration_timepoints"]
                solve_time[len(devices) * k + d_no, t] = np.array([item for item in metrics["solve_time_breakdown"].values()])

print(f"Done! Generating plots...")

# ------------ plots over solver iteration ----------------
args.downsample_str = "-ds" if args.downsample else ""
binary_str = "2" if args.binary_classes else "10"
dtime = dtime.replace("/", "").replace(" ","").replace(":", "")
base_save_str = f"cif{binary_str}{args.downsample_str}-n={args.n_train}-loss={args.loss_type}-P={args.P_S}-{args.backend}-{dtime}"
x_vals = solve_timepoints

# with all types of log scaling
for x_logscale in [True, False]:
    for y_logscale in [True, False]:
        plt.figure(figsize=(10,8), dpi=300)
        plt.subplot(221)
        plot_metric(train_loss, x_vals, optimizer_labs, 
                    x_logscale=x_logscale,
                    y_logscale=y_logscale,
                    title_str = '(a) Train Loss')

        plt.subplot(222)
        plot_metric(train_acc, x_vals, optimizer_labs,
                    x_logscale=x_logscale,
                    y_logscale=y_logscale, 
                    title_str = '(b) Train Accuracy')

        plt.subplot(223)
        plot_metric(test_loss, x_vals, optimizer_labs, 
                    x_logscale=x_logscale,
                    y_logscale=y_logscale,
                    title_str = '(c) Test Loss',
                    xlabel_str = "Solve time (s)")

        plt.subplot(224)
        plot_metric(test_acc, x_vals, optimizer_labs, 
                    x_logscale=x_logscale,
                    y_logscale=y_logscale,
                    title_str = '(d) Test Accuracy',
                    xlabel_str = "Solve time (s)",
                    use_legend=True)
        plt.suptitle(f"CIFAR-10 Performance vs. Solve Time on CPU at downsampled scale")
        plt.savefig(os.path.join(args.save_root, f"performance-{base_save_str}_xlog={int(x_logscale)}_ylog={int(y_logscale)}.png"))

# save solve time in datatable
solve_times_mean = np.mean(solve_time, axis=1)
solve_data = {
    'total_time': solve_times_mean[:,0],
    'precomputation': solve_times_mean[:,1],
    'u updates': solve_times_mean[:,2],
    'v updates': solve_times_mean[:,3],
    's updates': solve_times_mean[:,4],
    'dual updates': solve_times_mean[:,5],
}
solve_df = pd.DataFrame(solve_data, index = optimizer_labs)
solve_df.to_csv(os.path.join(args.save_root, "times-"+base_save_str+".csv"))
print("Final solve times: ")
print(solve_df)

# plot solvetime bars in stack manner
plt.figure(figsize=(9,6), dpi=300)
precomps = solve_times_mean[:,1]
primal_updates = solve_times_mean[:,2]
primal_updates2 = solve_times_mean[:,3] + solve_times_mean[:,4]
dual_updates = solve_times_mean[:,5]
alph = 0.8
plt.bar(optimizer_labs, dual_updates, bottom=primal_updates+primal_updates2+precomps, color = "saddlebrown", alpha = alph, label='dual updates')
plt.bar(optimizer_labs, primal_updates2, bottom=precomps+primal_updates, color = "lightcoral", alpha = alph, label='v updates')
plt.bar(optimizer_labs, primal_updates, bottom=precomps, color = "darkred", alpha = alph, label='u updates')
plt.bar(optimizer_labs, precomps, color = "black", alpha = alph, label="precomps")
plt.title(f"Mean runtime")
plt.ylabel("Time (s)")
plt.legend(facecolor='white',frameon = True)
plt.savefig(os.path.join(args.save_root, "times-"+base_save_str+".png"))

plt.show()

