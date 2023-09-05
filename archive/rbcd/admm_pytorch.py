# -*- coding: utf-8 -*-
"""ADMM PyTorch runner 
"""

# Figures
import matplotlib.pyplot as plt

# Performance measurement
from time import perf_counter

import numpy as np
import torch

import random
my_seed = 20220518
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Helper functions
from rbcd.utils import *
from utils.load_data import load_mnist, load_fmnist, load_cifar
from rbcd.postprocess import *
from rbcd.ADMM_torch import ADMMTrainer

"""## MNIST

### Binary Cross-Entropy
"""

beta, P, n = .0001, 8, 12000

#X, y, X_test, y_test = load_fmnist(n=n, downsample=False)
X, y, X_test, y_test = load_mnist(n=n, downsample=True)
yy, yy_test = ((y+1)//2).astype(int), ((y_test+1)//2).astype(int)

# Show dims
print(f'X_train shape = {X.shape}')
print(f'X_test  shape = {X_test.shape}')
print(f'y_train shape = {yy.shape}')
print(f'y_test  shape = {yy_test.shape}')


print(yy[:40])
print(yy_test[:40])


dmat, n, d, P, v, w = generate_D(X, P)

"""### ADMM-RBCD"""

# ADMM
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)
t2 = perf_counter()

runs, iters = 5, 35
cost, accuracy_train, accuracy_test = np.empty(runs), np.empty(runs), np.empty(runs)

for r in range(runs):
    print("Run", r+1)
    admm_rbcd_trainer = ADMMTrainer(
        X, yy, P=P, beta=beta, rho=.02, gamma_ratio=.2, alpha0=3e-6, dmat=None, loss_type='ce', 
        X_test=X_test, y_test=yy_test, iters=iters, RBCDthresh=.7, RBCD_block_size=3)
    
    costs, costs2, dists, accuracies, v, w, u, alpha = admm_rbcd_trainer.ADMM_train()

    print("Evaluating on training set...")
    accuracies_train, yhat_train = evaluate(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=True)
    print("Evaluating on test set...")
    accuracies_test,  yhat_test  = evaluate(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=True)
    
    cost[r] = np.min(costs2)
    accuracy_train[r] = np.max(accuracies_train)
    accuracy_test[r] = np.maximum(np.max(accuracies_test), np.max(accuracies))

print("\n***********************")
print('Average training accuracy:', np.mean(accuracy_train),
      ', Average test accuracy:', np.mean(accuracy_test),
      ', Average loss:', np.mean(cost) )
print('Median training accuracy:', np.median(accuracy_train),
      ', Median test accuracy:', np.median(accuracy_test),
      ', Median loss:', np.median(cost))
print('Std div accuracy:', np.std(accuracy_train), ', Std div loss:', np.std(cost)  )
print("Average time:", (perf_counter() - t2) / runs, 'seconds.')
