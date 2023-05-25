import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from keras.datasets import mnist
from relu_solver import Approximate_2_Layer_ReLU
from relu_utils import squared_loss, binary_classifcation_accuracy
from optimizers import admm_optimizer, approx_admm_optimizer
from torch.utils.data import DataLoader, TensorDataset

# Params
m = 8
P_S = m
rho  = 0.001
step = 0.001
beta = 0.0001
bias = True
seed = 364
standardize_data = False
max_iter = 6

print(f'loading data...')

# ------------ Load Data ------------
# Load mnist and select only digts 2 and 8 (according to paper ...)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_size = X_train.shape[1:]
inds_train = np.argwhere(np.bitwise_or(y_train == 2, y_train==8))[:, 0]
inds_test  = np.argwhere(np.bitwise_or(y_test == 2, y_test==8))[:, 0]
X_train, X_test = X_train[inds_train], X_test[inds_test]
y_train, y_test = 1 - 2.0 * (y_train[inds_train] > 4), 1 - 2.0 * (y_test[inds_test] > 4)

# Paper uses d = 100, which I assume they do by downsampling
X_train = X_train.reshape((X_train.shape[0], -1)).astype(float)[:, ::7]
X_test = X_test.reshape((X_test.shape[0], -1)).astype(float)[:, ::7]

# Show dims
print(f'X_train shape = {X_train.shape}')
print(f'X_test  shape = {X_test.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'y_test  shape = {y_test.shape}')

params = dict(m=m,
             P_S=m,
             rho=rho,
             step=step,
             beta=beta,
             bias=bias,
             seed=seed,
             standardize_data=standardize_data,
             acc_func=binary_classifcation_accuracy,
)

# ------------ Train Approximate (RBCD) ADMM ------------

# # using approximate admm solver
# solver = Approximate_2_Layer_ReLU(**params, optimizer=approx_admm_optimizer)

# solver.optimize(X_train, y_train, max_iter=10, verbose=True)

# print("\nAPROX ADMM SOLVER PERFORMANCE:")
# y_hat_train = solver.predict(X_train, weights="C-ReLU")
# y_hat_test = solver.predict(X_test, weights="C-ReLU")
# print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
# print(f"Train accuracy: {binary_classifcation_accuracy(y_hat_train, y_train)}")
# print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
# print(f"Test accuracy: {binary_classifcation_accuracy(y_hat_test, y_test)}")

# ------------ Train ADMM ------------

# using admm solver
solver = Approximate_2_Layer_ReLU(**params, optimizer=admm_optimizer)

solver.optimize(X_train, y_train, max_iter=max_iter, verbose=True)

print("\nADMM SOLVER PERFORMANCE:")
y_hat_train = solver.predict(X_train, weights="C-ReLU")
y_hat_test = solver.predict(X_test, weights="C-ReLU")
print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
print(f"Train accuracy: {binary_classifcation_accuracy(y_hat_train, y_train)}")
print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
print(f"Test accuracy: {binary_classifcation_accuracy(y_hat_test, y_test)}")



# ------------ Train MLP ------------
# Try to use GPU 
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training params
lr = 1e-3
batch_size = 2 ** 8
epochs = 50

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

# Prediction
with torch.no_grad():
    y_hat_train = MLP(X_train_torch.to(dev)).to('cpu').numpy()
    y_hat_test = MLP(X_test_torch.to(dev)).to('cpu').numpy()

# Show results
print("\nNONCONVEX PROBLEM WEIGHTS:")
print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
print(f"Train accuracy: {binary_classifcation_accuracy(y_hat_train, y_train)}")
print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
print(f"Test accuracy: {binary_classifcation_accuracy(y_hat_test, y_test)}")

# Compare both methods
plt.subplot(211)
plt.title('CVX Training Accuracy')
plt.plot(solver.metrics["train_acc"])
plt.subplot(212)
plt.title('MLP Training Accuracy')
plt.plot(train_accs)
plt.show()