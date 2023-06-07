import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from utils.load_data import load_mnist

from scnn.models import ConvexGatedReLU, ConvexReLU
from scnn.solvers import RFISTA, AL, LeastSquaresSolver, CVXPYSolver, ApproximateConeDecomposition
from scnn.regularizers import NeuronGL1, L2, L1
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.optimize import optimize_model, optimize
from scnn.private.utils.data import gen_classification_data

"""################ PARAMETERS FOR DATA GENERATION ###################"""
# ----------- Data parameters ------------
dataset_path = "baADMM/datasets/mnist.pkl.gz"
downsample = True # downsample data dim to 100 if True


# Generate realizable synthetic classification problem (ie. Figure 1)
n_train = 10000
n_test = 1000
max_neurons = 500

# ------------ Load Data ------------
print(f'Loading data...')
# Load mnist and select only digts 2 and 8, but only get 1000 samples
X_train, y_train, X_test, y_test = load_mnist(dataset_rel_path=dataset_path, n=n_train, downsample=downsample)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

X_test = X_test[:n_test,:]
y_test = y_test[:n_test,:]

regularizer = NeuronGL1(0.01)
metrics = Metrics(metric_freq=25, model_loss=True, train_accuracy=True, train_mse=True, test_mse=True, test_accuracy=True, neuron_sparsity=True)

G = sample_gate_vectors(123, X_train.shape[1], max_neurons)

model = ConvexGatedReLU(G)
solver = RFISTA(model, tol=1e-6)

grelu_model, grelu_metrics = optimize_model(
    model,
    solver,
    metrics,
    X_train, 
    y_train,
    X_test, 
    y_test,
    regularizer=regularizer,
    verbose=True,
)

print(grelu_metrics.objective)
print(grelu_metrics.time)
print(grelu_metrics.train_accuracy)
print(grelu_metrics.test_accuracy)