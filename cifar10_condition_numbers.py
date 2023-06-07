"""
Test performance of all ADMM varients on CIFAR-10 with any combination of compute args.backends 

Script generates:
- condition numbers for matrix A under different preconditioners
"""

import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import argparse
from utils.load_data import load_cifar
from datetime import datetime
import os
dtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

from utils.admm_utils import FG_Operators, get_hyperplane_cuts, nystrom_sketch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADMM methods on CIFAR-10."
    )
    
    parser.add_argument("--dataset_path", type=str, default="baADMM/datasets/cifar-10-batches-py") # path to cifar model
    parser.add_argument("--n_train", type=int, default=1000) # fixed number of training examples to use
    parser.add_argument("--n_test", type=int, default=1000) # fixed number of training examples to use
    parser.add_argument("--ntrials", type=int, default=5) # number of trials

    parser.set_defaults(downsample=True)
    parser.add_argument('--no_downsample', dest='downsample', action='store_false', default=True) # downsample data dim to 100 if True
    parser.add_argument("--P_S", type=int, default=10) # number of sampled hyperplanes
    parser.add_argument("--save_root", type=str, default="figures/cifar-10") # where to save figures to

    args = parser.parse_args()
    return args

args = parse_args()
P_S = args.P_S

# ------------ Load Data ------------
print(f'Loading data...')
# Load mnist and select only digts 2 and 8, but only get 1000 samples
os.makedirs(args.save_root, exist_ok=True)
X_train, y_train, _, _ = load_cifar(dataset_rel_path=args.dataset_path, n=args.n_train, downsample=args.downsample)
X = np.hstack([X_train, np.ones((X_train.shape[0],1))])

print("Computing A...")

# construct d-diags
d_diags = get_hyperplane_cuts(X, P_S, seed=1)
OPS = FG_Operators(d_diags, X, rho=0.001)
n, d = X.shape
# construct A
A = np.eye(2 * d * P_S)
for i in range(P_S):
    for j in range(P_S):
        # perform multiplication 
        FiFj = OPS.F(i % P_S).T @ OPS.F(j % P_S) / OPS.rho
        A[i*d:(i+1)*d, j*d:(j+1)*d] += FiFj
        A[(i+P_S)*d:(i+P_S+1)*d, (j)*d:(j+1)*d] += - FiFj
        A[(i)*d:(i+1)*d, (j+P_S)*d:(j+P_S+1)*d] += - FiFj
        A[(i+P_S)*d:(i+P_S+1)*d, (j+P_S)*d:(j+P_S+1)*d] += FiFj

# print raw condition number 
print("Computing k(A)...")

kA = np.linalg.cond(A)

# condition number preconditioned by diagonal 
print("Computing k(JA)...")

Jacobi = 1 / np.diag(A)
kJ = np.linalg.cond(Jacobi * A)

# condition number with nystrom sketch 
print("Computing k(NA)...")

U, S = nystrom_sketch(A, rank=10)
rho = 1
kN = np.linalg.cond((((S[-1] + rho) * (U / (S + rho)) @ U.T) + (np.eye(U.shape[0]) - U @ U.T)) @ A)


print(f"Condition number of A: {kA}")
print(f"Condition number of jacobi precond A: {kJ}")
print(f"Condition number of nys precond A: {kN}")