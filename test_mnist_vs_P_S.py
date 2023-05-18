import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

from keras.datasets import mnist
from relu_solver import Approximate_2_Layer_ReLU
from relu_utils import squared_loss, binary_classifcation_accuracy
from optimizers import admm_optimizer, approx_admm_optimizer, cvxpy_optimizer
from torch.utils.data import DataLoader, TensorDataset

from ADMM_torch import ADMMTrainer
from postprocess import evaluate
from load_data import load_mnist

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
    y_test_torch = torch.as_tensor(y_test[:, None], dtype=torch.float32)
    trainset = TensorDataset(X_train_torch, y_train_torch)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss
    optim = torch.optim.Adam(MLP.parameters(), lr=lr)
    loss_func = nn.MSELoss()

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

# ------------ Train Approx ADMM ----------
def run_admm_approx(X_train, y_train, X_test, y_test, P):

    admm_rbcd_trainer = ADMMTrainer(
    X_train, y_train, P=P, beta=.0001, rho=.02, gamma_ratio=.2, alpha0=3e-6, dmat=None, loss_type='ce', 
    X_test=X_test, y_test=y_test, iters=35, RBCDthresh=.7, RBCD_block_size=3)

    start_time = time.perf_counter()
    _, costs2, _, accuracies, _, _, u, alpha = admm_rbcd_trainer.ADMM_train()
    solve_time = time.perf_counter() - start_time
    accuracies_train, _ = evaluate(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=False)
    accuracies_test,  _  = evaluate(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=False)
    
    train_loss = np.min(costs2)
    train_acc = np.max(accuracies_train)
    test_acc = np.maximum(np.max(accuracies_test), np.max(accuracies))
    
    return train_loss, train_acc, test_acc, solve_time
    

# ----------- PARAMETERS ------------
rho  = 0.01
step = 0.01
beta = 0.001
bias = True
seed = 364
standardize_data = False
max_iter = 10

ntrials = 5

optimizers = [admm_optimizer, cvxpy_optimizer]
optimizer_labs = ["ADMM", "CVXPY"]

print(f'loading data...')

# ------------ Load Data ------------
# Load mnist and select only digts 2 and 8, but only get 1000 samples
X_train, y_train, X_test, y_test = load_mnist(n=1000, downsample=True)
yy_train, yy_test = ((y_train+1)//2).astype(int), ((y_test+1)//2).astype(int)


# Show dims
print(f'X_train shape = {X_train.shape}')
print(f'X_test  shape = {X_test.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'y_test  shape = {y_test.shape}')

# ------------ Figure 2: Accuracy and time vs. P_S ------------
# values of n to try
PS_vals = np.array([4,10,20,30,40,50])
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
        try:
            for (k, optimizer) in enumerate(optimizers):
                if valid_optimizers[k]:
                    print(f"  {optimizer_labs[k]} trial {t+1}/{ntrials}")
                    train_loss[k, i, t], _, train_acc[k,i, t], test_acc[k,i,t], solve_time[k, i,t] = run_admm(X_train,
                                                                                                            y_train,
                                                                                                            X_test,
                                                                                                            y_test,
                                                                                                            params,
                                                                                                            optimizer,
                                                                                                            max_iter)
        except:
            print(f"Solve failed for {optimizer_labs[k]} trial {t+1}/{ntrials}")
            train_loss[k, i, :] = np.nan
            train_acc[k,i, :] = np.nan
            test_acc[k,i,:] = np.nan
            solve_time[k,i,:] = np.nan
            valid_optimizers[k] = 0

        # run approx admm
        print(f"  admm-approx trial {t+1}/{ntrials}")
        train_loss[-2, i,t], train_acc[-2,i,t], test_acc[-2,i,t], solve_time[-2, i,t] = run_admm_approx(X_train, yy_train, X_test, yy_test, P=P_S)

        # run mlp 
        print(f"  mlp-nn trial {t+1}/{ntrials}")
        train_loss[-1, i,t], _, train_acc[-1,i,t], test_acc[-1,i,t], solve_time[-1, i,t] = run_nn_mlp(X_train, y_train, X_test, y_test, P_S, batch_size=64)


print(f"Done! Generating plots...")
# average across trials
train_loss = np.mean(train_loss, axis=2)
train_acc = np.mean(train_acc, axis=2)
test_acc = np.mean(test_acc, axis=2)
solve_time = np.mean(solve_time, axis=2)

# ------------ plots for varied P_S ----------------
plt.subplot(221)
plt.title('(a) Train Loss vs. P_S')
plt.plot(PS_vals, train_loss[-2], '--o', label="ADMM-RBCD")
for k, lab in enumerate(optimizer_labs):
    plt.plot(PS_vals, train_loss[k], '--o', label=lab)
plt.plot(PS_vals, train_loss[-1], '--o', label="MLP-NN")
plt.legend()

plt.subplot(222)
plt.title('(b) Solve Time vs. P_S')
plt.plot(PS_vals, solve_time[-2], '--o', label="ADMM-RBCD")
for k, lab in enumerate(optimizer_labs):
    plt.plot(PS_vals, solve_time[k], '--o', label=lab)
plt.plot(PS_vals, solve_time[-1], '--o', label="MLP-NN")
plt.ylabel("CPU Time (s)")
plt.legend()

plt.subplot(223)
plt.title('(c) Train Accuracy vs. P_S')
plt.plot(PS_vals, train_acc[-2], '--o', label="ADMM-RBCD")
for k, lab in enumerate(optimizer_labs):
    plt.plot(PS_vals, train_acc[k], '--o', label=lab)
plt.plot(PS_vals, train_acc[-1], '--o', label="MLP-NN")
plt.xlabel("Number of hidden layers P_S"); 
plt.legend()

plt.subplot(224)
plt.title('(d) Test Accuracy vs. P_S')
plt.plot(PS_vals, test_acc[-2], '--o', label="ADMM-RBCD")
for k, lab in enumerate(optimizer_labs):
    plt.plot(PS_vals, test_acc[k], '--o', label=lab)
plt.plot(PS_vals, test_acc[-1], '--o', label="MLP-NN")
plt.xlabel("Number of hidden layers P_S"); 
plt.legend()

plt.show()
