
# Need to clean up 
# Miria: Friday night due, fix runner

# Figures
import matplotlib.pyplot as plt

# Performance measurement
from time import perf_counter

import numpy as np
import torch

# Reproduce
import random
my_seed = 20220518
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Helper functions
from utils import *
from load_data import load_mnist, load_fmnist, load_cifar
from postprocess import *
from ADMM_torch import ADMMTrainer

# ##  Fashion MNIST

# ### Binary Cross-Entropy

beta, P, n = .0001, 72, 12000
X, y, X_test, y_test = load_fmnist(n=n, downsample=False)
yy, yy_test = ((y+1)//2).astype(int), ((y_test+1)//2).astype(int)
dmat, n, d, P, v, w = generate_D(X, P)

# ### ADMM-RBCD

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

# ## CIFAR-10

# ### Binary Cross-Entropy

beta, P, n = .0002, 120, 10000
X, y, X_test, y_test = load_cifar(n=n)
yy, yy_test = ((y+1)//2).astype(int), ((y_test+1)//2).astype(int)
dmat, n, d, P, v, w = generate_D(X, P)

# ### ADMM-RBCD

# ADMM
t2 = perf_counter()
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

runs, iters = 5, 30
cost, accuracy_train, accuracy_test = np.empty(runs), np.empty(runs), np.empty(runs)

for r in range(runs):
    print("Run", r+1)
    admm_rbcd_trainer = ADMMTrainer(
        X, yy, P=P, beta=beta, rho=.1, gamma_ratio=.25, alpha0=2e-6, dmat=None, loss_type='ce', 
        X_test=X_test, y_test=yy_test, iters=iters, RBCDthresh=.85, RBCDthresh_decay=.95, RBCD_block_size=4)
    costs, costs2, dists, accuracies, v, w, u, alpha = admm_rbcd_trainer.ADMM_train()

    print("Evaluating on training set...")
    accuracies, yhat = evaluate(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=True)
    accuracies2, yhat2 = evaluate2(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=True)
    print("Evaluating on test set...")
    accuracies_test,  yhat_test  = evaluate(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=True)
    accuracies_test2, yhat_test2 = evaluate2(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=True)
    
    cost[r] = np.min(costs2)
    accuracy_train[r] = np.maximum(np.max(accuracies), np.max(accuracies2))
    accuracy_test[r] = np.maximum(np.max(accuracies_test), np.max(accuracies_test2))

print("\n***********************")
print('Average training accuracy:', np.mean(accuracy_train),
      ', Average test accuracy:', np.mean(accuracy_test),
      ', Average loss:', np.mean(cost) )
print('Median training accuracy:', np.median(accuracy_train),
      ', Median test accuracy:', np.median(accuracy_test),
      ', Median loss:', np.median(cost))
print('Std div accuracy:', np.std(accuracy_train), ', Std div loss:', np.std(cost))
print("Average time:", (perf_counter() - t2) / runs, 'seconds.')

# ### ADMM-RBCD GPU

beta, P, n = .0002, 120, 10000
X, y, X_test, y_test = load_cifar(n=n)
yy, yy_test = ((y+1)//2).astype(int), ((y_test+1)//2).astype(int)
dmat, n, d, P, v, w = generate_D(X, P)

# ADMM
t2 = perf_counter()

runs, iters = 5, 30
cost, accuracy_train, accuracy_test = np.empty(runs), np.empty(runs), np.empty(runs)

for r in range(runs):
    print("Run", r+1)
    admm_rbcd_trainer = ADMMTrainer(
        X, yy, P=P, beta=beta, rho=.1, gamma_ratio=.25, alpha0=2e-6, dmat=None, loss_type='ce',
        X_test=X_test, y_test=yy_test, iters=iters, RBCDthresh=.85, RBCDthresh_decay=.95, RBCD_block_size=4)
    costs, costs2, dists, accuracies, v, w, u, alpha = admm_rbcd_trainer.ADMM_train(RBCD_verbose=False)

    print("Evaluating on training set...")
    accuracies, yhat = evaluate(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=True)
    accuracies2, yhat2 = evaluate2(
        admm_rbcd_trainer.X, admm_rbcd_trainer.y, u, alpha, use_torch=True, verbose=True)
    print("Evaluating on test set...")
    accuracies_test,  yhat_test  = evaluate(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=True)
    accuracies_test2, yhat_test2 = evaluate2(
        admm_rbcd_trainer.X_test, admm_rbcd_trainer.y_test, u, alpha, use_torch=True, verbose=True)
    
    cost[r] = np.min(costs2)
    accuracy_train[r] = np.maximum(np.max(accuracies), np.max(accuracies2))
    accuracy_test[r] = np.maximum(np.max(accuracies_test), np.max(accuracies_test2))

print("\n***********************")
print('Average training accuracy:', np.mean(accuracy_train),
      ', Average test accuracy:', np.mean(accuracy_test),
      ', Average loss:', np.mean(cost) )
print('Median training accuracy:', np.median(accuracy_train),
      ', Median test accuracy:', np.median(accuracy_test),
      ', Median loss:', np.median(cost))
print('Std div accuracy:', np.std(accuracy_train), ', Std div loss:', np.std(cost))
print("Average time:", (perf_counter() - t2) / runs, 'seconds.')


