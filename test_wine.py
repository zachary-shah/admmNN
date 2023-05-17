import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from relu_solver import Approximate_2_Layer_ReLU
from relu_utils import squared_loss, classifcation_accuracy
from optimizers import admm_optimizer, cvxpy_optimizer

# generate toy data trying to fit noisy data to cosine
data = pd.read_csv("test_data/winequality-red.csv", delimiter=";")
X = np.array(data)[:,:11]
y = np.array(data.quality)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_train, d = X_train.shape
n_test = X_test.shape[0]

print(f"n train = {n_train}")
print(f"n test = {n_test}")

m = 10

# SOLVE THE CONVEX PROBLEM

params = dict(m = 10,
            P_S = 20,
            rho = 0.0001,
            step = 0.00001,
            beta = 0.0001,
            bias = True,
            seed = 364,
            standardize_data = False,
)
max_iter = 10

# using admm solver
solver = Approximate_2_Layer_ReLU(**params, optimizer=admm_optimizer)

solver.optimize(X_train, y_train, max_iter=max_iter, verbose=False)

print("ADMM SOLVER PERFORMANCE:")
y_hat_train = solver.predict(X_train, weights="C-ReLU")
y_hat_test = solver.predict(X_test, weights="C-ReLU")
print(f"Train loss: {squared_loss(y_hat_train, y_train)}")
print(f"Train accuracy: {classifcation_accuracy(y_hat_train, y_train)}")
print(f"Test loss: {squared_loss(y_hat_test, y_test)}")
print(f"Test accuracy: {classifcation_accuracy(y_hat_test, y_test)}")
