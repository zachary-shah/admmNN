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
optimizers = [admm_optimizer, cvxpy_optimizer] # list of functions for optimizers to run
optimizer_labs = ["ADMM", "CVXPY"] # string labels for each optimizer

# ----------- Figure 1: data varied with P_S params ------------
PS_vals = np.array([4,10,20,30,40,50]) # the P_S values to test against
n_for_varied_P_S = 1000 # fixed number of training examples to use

# ----------- Figure 2: data varied with n params ------------
number_of_nvals = 9 # number of n values to test between min_n and max_n
P_S_for_varied_n = 8 # fixed P_S to use for varied n trials
min_n = 100 # lower end of n to use (>1)
max_n = np.infty # max n to test (or limited by max n in data)

"""#################################################################"""

# ------------ Train ADMM ----------
def run_admm(X_train, y_train, X_test, y_test, params, optimizer, max_iter):
    # using approximate admm solver
    solver = Approximate_2_Layer_ReLU(**params, optimizer=optimizer)

    solver.optimize(X_train, y_train, max_iter=max_iter, verbose=False)

    y_hat_train = solver.predict(X_train, weights="C-ReLU")
    y_hat_test = solver.predict(X_test, weights="C-ReLU")
    train_loss = squared_loss(y_hat_train, y_train)
    test_loss = squared_loss(y_hat_test, y_test)
    train_acc = binary_classifcation_accuracy(y_hat_train, y_train)
    test_acc = binary_classifcation_accuracy(y_hat_test, y_test)

    return train_loss, test_loss, train_acc, test_acc, solver.metrics["solve_time"]

# ------------ Train MLP ------------
def run_nn_mlp(X_train, y_train, X_test, y_test, m=8,
              dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              lr = 1e-4,
              batch_size = 2 ** 8,
              epochs = 50, ):

    # Network
    n, d = X_train.shape
    MLP = nn.Sequential(nn.Linear(d, m),
                        nn.ReLU(),
                        nn.Linear(m, 1)).to(dev)

    # Train/Test data to torch
    X_train_torch = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.as_tensor(y_train[:, None], dtype=torch.float32)
    X_test_torch = torch.as_tensor(X_test, dtype=torch.float32)
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
    train_accs = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}', end='\r', flush=True)
        for sample, label in iter(trainloader):

            # Move to device
            sample = sample.to(dev)
            label = label.to(dev)

            # predict 
            pred = MLP(sample)

            # Back prop
            loss = loss_func(pred, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_accs.append(binary_classifcation_accuracy(
                pred.to('cpu').detach().numpy(),
                label.to('cpu').detach().numpy()))
            
    solve_time = time.perf_counter() - start

    # Prediction
    with torch.no_grad():
        y_hat_train = MLP(X_train_torch.to(dev)).to('cpu').numpy()
        y_hat_test = MLP(X_test_torch.to(dev)).to('cpu').numpy()

    train_loss = squared_loss(y_hat_train, y_train)
    test_loss = squared_loss(y_hat_test, y_test)
    train_acc = binary_classifcation_accuracy(y_hat_train, y_train)
    test_acc = binary_classifcation_accuracy(y_hat_test, y_test)

    return train_loss, test_loss, train_acc, test_acc, solve_time

# ------------ Train Approx ADMM with Miria's solver ----------
def run_admm_approx(X_train, y_train, X_test, y_test, P, loss_type):

    admm_rbcd_trainer = ADMMTrainer(
    X_train, y_train, P=P, beta=.0001, rho=.02, gamma_ratio=.2, alpha0=3e-6, dmat=None, loss_type=loss_type, 
    X_test=X_test, y_test=y_test, iters=35, RBCDthresh=.7, RBCD_block_size=3)

    start_time = time.perf_counter()
    _, costs2, _, accuracies, _, _, u, alpha = admm_rbcd_trainer.ADMM_train(verbose=False, RBCD_verbose=False)
    solve_time = time.perf_counter() - start_time
    accuracies_train, _ = evaluate(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=False)
    accuracies_test,  _  = evaluate(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=False)
    
    train_loss = np.min(costs2)
    train_acc = np.max(accuracies_train)
    test_acc = np.maximum(np.max(accuracies_test), np.max(accuracies))
    
    return train_loss, train_acc, test_acc, solve_time
    
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

    full_labs = optimizer_labs + ["ADMM-RBCD", "MLP-NN"]

    for indx, lab in enumerate(full_labs):
        upper_error = np.minimum(metric_max[indx], metric_mean[indx] + metric_std[indx])
        lower_error = np.maximum(metric_min[indx], metric_mean[indx] - metric_std[indx])

        plt.plot(x_vals, metric_mean[indx], '--o', label=lab)
        plt.fill_between(x_vals, lower_error, upper_error, alpha=0.2)

    plt.title(title_str)
    plt.xlabel(xlabel_str) 
    plt.ylabel(ylabel_str)
    if use_legend: plt.legend()

# ------------ Print configs for log ------------
dtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"""
TESTING RELU ON MNIST! 
EXPERIMENT DATE: {dtime}...
  MNIST Data Params:
    Downsampled from d=784 to 100: {downsample}
    standardized data: {standardize_data}
  Experiment Params:
    ntrials: {ntrials}
    loss type: {loss_type}
  ADMM Solver params:
    rho = {rho}
    step = {step}
    beta = {beta}
    bias = {bias}
    max_iter = {max_iter}
  Optimizers used:
    Defaults: MLP-NN, ADMM-RBCD
    Additional: {optimizer_labs}
  Figure 1 params: 
    PS_vals: {PS_vals}
    n: {n_for_varied_P_S}
  Figure 2 params:
    n vals: {number_of_nvals} between {min_n} and {max_n}
    PS: {P_S_for_varied_n}\n
""")


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
nopt = len(optimizers) + 2
valid_optimizers = np.ones((len(optimizers),))
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
        try:
            for (k, optimizer) in enumerate(optimizers):
                if valid_optimizers[k]:
                    print(f"  {optimizer_labs[k]} trial {t+1}/{ntrials}")
                    train_loss[k, i, t], _, train_acc[k,i, t], test_acc[k,i,t], solve_time[k, i,t] = run_admm(X_tr, y_tr,X_test,y_test,params,optimizer,max_iter)
        except:
            print(f"Solve failed for {optimizer_labs[k]} trial {t+1}/{ntrials}")
            train_loss[k, i, :], train_acc[k,i, :], test_acc[k,i,:], solve_time[k,i,:]  = np.nan, np.nan, np.nan, np.nan
            valid_optimizers[k] = 0

        # run approx admm. use [0 1] labels for CE and [-1 1] for MSE
        print(f"  admm-approx trial {t+1}/{ntrials}") 
        if loss_type == "ce":
            train_loss[-2, i,t], train_acc[-2,i,t], test_acc[-2,i,t], solve_time[-2, i,t] = run_admm_approx(X_tr, yy_tr, X_test, yy_test, P=P_S, loss_type=loss_type)
        else: 
            train_loss[-2, i,t], train_acc[-2,i,t], test_acc[-2,i,t], solve_time[-2, i,t] = run_admm_approx(X_tr, y_tr, X_test, y_test, P=P_S, loss_type=loss_type)

        # run mlp 
        print(f"  mlp-nn trial {t+1}/{ntrials}")
        train_loss[-1, i,t], _, train_acc[-1,i,t], test_acc[-1,i,t], solve_time[-1, i,t] = run_nn_mlp(X_tr, y_tr, X_test, y_test, P_S, batch_size=64)


print(f"Done! Generating plots...")

# ------------ plots for varied P_S ----------------
plt.subplot(221)
plot_metric(train_loss, PS_vals, optimizer_labs, 
            title_str = '(a) Train Loss vs. P_S')

plt.subplot(222)
plot_metric(solve_time, PS_vals, optimizer_labs, 
            title_str = '(b) Solve Time vs. P_S',
            ylabel_str = "CPU Time (s)")

plt.subplot(223)
plot_metric(train_acc, PS_vals, optimizer_labs, 
            title_str = '(c) Train Accuracy vs. P_S',
            xlabel_str = "Number of hidden layers P_S")

plt.subplot(224)
plot_metric(test_acc, PS_vals, optimizer_labs, 
            title_str = '(d) Test Accuracy vs. P_S',
            xlabel_str = "Number of hidden layers P_S",
            use_legend=True)

plt.suptitle(f"Figure 1. Performance vs. P_S for loss type: {loss_type}")

# ------------ Figure 2: Accuracy and time vs. n ------------
print("Generating figure 2 data...")
nopt = len(optimizers) + 2
nvals = np.linspace(max(min_n, 1), min(X_train.shape[0], max_n), number_of_nvals).astype(int)
valid_optimizers = np.ones((len(optimizers),))
train_loss = np.zeros((nopt, len(nvals),ntrials)) * np.nan
train_acc = np.zeros((nopt, len(nvals),ntrials)) * np.nan
test_acc = np.zeros((nopt, len(nvals),ntrials)) * np.nan
solve_time = np.zeros((nopt, len(nvals),ntrials)) * np.nan

params = dict(m=P_S_for_varied_n,
             P_S=P_S_for_varied_n,
             rho=rho,
             step=step,
             beta=beta,
             bias=bias,
             standardize_data=standardize_data,
             acc_func=binary_classifcation_accuracy,
)

for (i,n) in enumerate(nvals):

    X_tr = X_train[:n, :]
    y_tr = y_train[:n]
    yy_tr = yy_train[:n]

    print(f"TRIALS FOR n={n} (size {i+1}/{len(nvals)})")
    print(f"  Proportion of 8s in train data: {np.sum(y_tr == 1)/len(y_tr)}")

    for t in range(ntrials):
        
        # run for remaining valid optimizers (because cvxpy fails for high data dimension, so don't continue with cvxpy after it fails once)
        try:
            for (k, optimizer) in enumerate(optimizers):
                if valid_optimizers[k]:
                    print(f"  {optimizer_labs[k]} trial {t+1}/{ntrials}")
                    train_loss[k, i, t], _, train_acc[k,i, t], test_acc[k,i,t], solve_time[k, i,t] = run_admm(X_tr,y_tr,X_test,y_test,params,optimizer,max_iter)
        except:
            print(f"Solve failed for {optimizer_labs[k]} trial {t+1}/{ntrials}")
            train_loss[k, i, :], train_acc[k,i, :], test_acc[k,i,:], solve_time[k,i,:]  = np.nan, np.nan, np.nan, np.nan
            valid_optimizers[k] = 0

        # run approx admm
        print(f"  admm-approx trial {t+1}/{ntrials}")
        if loss_type == "ce":
            train_loss[-2, i,t], train_acc[-2,i,t], test_acc[-2,i,t], solve_time[-2, i,t] = run_admm_approx(X_tr, yy_tr, X_test, yy_test, P=P_S_for_varied_n, loss_type=loss_type)
        else: 
            train_loss[-2, i,t], train_acc[-2,i,t], test_acc[-2,i,t], solve_time[-2, i,t] = run_admm_approx(X_tr, y_tr, X_test, y_test, P=P_S_for_varied_n, loss_type=loss_type)

        # run mlp 
        print(f"  mlp-nn trial {t+1}/{ntrials}")
        train_loss[-1, i,t], _, train_acc[-1,i,t], test_acc[-1,i,t], solve_time[-1, i,t] = run_nn_mlp(X_tr, y_tr, X_test, y_test, m=P_S_for_varied_n, batch_size=int(min(16,n//32)))

print(f"Done! Generating plots...")

# ------------ plots for varied n ----------------
plt.figure()
plt.subplot(221)
plot_metric(train_loss, nvals, optimizer_labs, 
            title_str = '(a) Train Loss vs. n')

plt.subplot(222)
plot_metric(solve_time, nvals, optimizer_labs, 
            title_str = '(b) Solve Time vs. n',
            ylabel_str = "CPU Time (s)")

plt.subplot(223)
plot_metric(train_acc, nvals, optimizer_labs, 
            title_str = '(c) Train Accuracy vs. n',
            xlabel_str = "Number of train examples n")

plt.subplot(224)
plot_metric(test_acc, nvals, optimizer_labs, 
            title_str = '(d) Test Accuracy vs. n',
            xlabel_str = "Number of train examples n",
            use_legend=True)

plt.suptitle(f"Figure 2. Performance vs. n for loss type: {loss_type}")
plt.show()