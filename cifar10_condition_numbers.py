"""
Test performance of all ADMM varients on CIFAR-10 with any combination of compute args.backends 

Script generates:
- condition numbers for matrix A under different preconditioners
"""

import argparse
import numpy as np
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
from utils.load_data import load_mnist, load_cifar
from datetime import datetime
import os
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
    parser.add_argument("--max_iter", type=int, default=10) # max number of admm iterations

    parser.set_defaults(downsample=True)
    parser.set_defaults(binary_classes=True)
    parser.add_argument('--no_downsample', dest='downsample', action='store_false', default=True) # downsample data dim to 100 if True
    parser.add_argument('--all_classes', dest='binary_classes', action='store_false') # decide if do 2 classes or 10 classes

    parser.add_argument("--standardize_data", action='store_true', default=False) # standardize data
    parser.add_argument("--P_S", type=int, default=10) # number of sampled hyperplanes
    parser.add_argument("--save_root", type=str, default="figures/cifar-10") # where to save figures to
    parser.add_argument("--backend", type=str, default="torch") # select compute backend
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

# ----------- Data parameters ------------

# ----------- Decide which optimizer methods to generate (at least one below must be "True") -----------
accuracy_func = binary_classifcation_accuracy if args.binary_classes else classifcation_accuracy
loss_func = cross_entropy_loss if args.loss_type=="ce" else squared_loss

devices = ["cpu", "cuda"]

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


# ------------ Load Data ------------
print(f'Loading data...')
# Load mnist and select only digts 2 and 8, but only get 1000 samples
os.makedirs(args.save_root, exist_ok=True)
X_train, y_train, _, _ = load_cifar(dataset_rel_path=args.dataset_path, n=args.n_train, downsample=args.downsample, binary_classes=args.binary_classes)


# construct d-diags
