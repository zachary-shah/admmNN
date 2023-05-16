import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from keras.datasets import mnist
from relu_solver import Approximate_2_Layer_ReLU
from relu_utils import squared_loss, classifcation_accuracy
from torch.utils.data import DataLoader, TensorDataset

# Params
m = 10
P_S = m
rho = 0.001
step = 0.00001
beta = 0.0001
bias = True
seed = 364
standardize_data = False
max_iter = 10

# ------------ Load Data ------------

# Load and flatten
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_size = X_train.shape[1:]
X_train = X_train[:1000, ...] / 255.0
X_test = X_test[:100, ...] / 255.0
y_train = y_train[:1000, ...].astype(float)
y_test = y_test[:100, ...].astype(float)
X_train = X_train.reshape((X_train.shape[0], -1)).astype(float)
X_test = X_test.reshape((X_test.shape[0], -1)).astype(float)

# Show dims
print(f'X_train shape = {X_train.shape}')
print(f'X_test  shape = {X_test.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'y_test  shape = {y_test.shape}')

# ------------ Train ADMM ------------
params = dict(m=m,
             P_S=m,
             rho=rho,
             step=step,
             beta=beta,
             bias=bias,
             seed=seed,
             standardize_data=standardize_data,
)

# using admm solver
solver = Approximate_2_Layer_ReLU(**params, optimizer="admm")

solver.optimize(X_train, y_train, max_iter=max_iter, verbose=True)

print("ADMM SOLVER PERFORMANCE:")
y_hat_train = solver.predict(X_train, weights="C-ReLU")
y_hat_test = solver.predict(X_test, weights="C-ReLU")
print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
print(f"Train accuracy: {classifcation_accuracy(y_hat_train, y_train)}")
print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
print(f"Test accuracy: {classifcation_accuracy(y_hat_test, y_test)}")

# # ------------ Train CVXPY ------------
# solver = Approximate_2_Layer_ReLU(**params, optimizer="cvxpy")

# solver.optimize(X_train, y_train, max_iter=max_iter, verbose=False)

# print("\nCVXPY SOLVER PERFORMANCE:")
# y_hat_train = solver.predict(X_train, weights="C-ReLU")
# y_hat_test = solver.predict(X_test, weights="C-ReLU")
# print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
# print(f"Train accuracy: {classifcation_accuracy(y_hat_train, y_train)}")
# print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
# print(f"Test accuracy: {classifcation_accuracy(y_hat_test, y_test)}")

# ------------ Train MLP ------------
