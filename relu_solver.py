"""
C-ReLU Optimizer Wrapper Class
"""

from time import perf_counter

from sklearn.preprocessing import StandardScaler

from optimizers import admm_optimizer, cvxpy_optimizer

from utils.relu_utils import squared_loss, cross_entropy_loss, classifcation_accuracy, optimal_weights_transform
from utils.admm_utils import ADMM_Params, OPTIM_MODES
from utils.typing_utils import ArrayType, EvalFunction, DATATYPE_BACKENDS, convert_backend_type

# abstract math library (wrapper around jnp.*, np.*, torch.* for all functions we need)
import utils.math_utils as mnp


"""
Shell structure for 2 Layer Convex ReLU MLP Solver
"""
class CReLU_MLP():
    """
    Set up solver
        :param X - training data with shape (n_train x d). Must be either np.ndarray or torch.tensor
        :param y - training labels with shape (n_train, ) or (n_train, 1). Must be either np.array, np.ndarray or torch.tensor
        :param P_S (Optional) - number of samples of ReLU activation patterns (D_i matrices)
        :param optimizer_mode (Optional) - string describing choice of optimizer, must be element of OPTIM_MODES
        :param loss_type (Optional) - string describing loss function in the form l(y_hat, y) that computes a loss, must be element of OPTIM_LOSSES
        :param datatype_backend (Optional) - string describing which data processing library to use in the backend: either numpy, torch, or jax.
        :param standardize_data - (Optional) True to standardize features using sclearn.preprocessing.StandardScaler 
        :param acc_func - (Optional) Specify a specific accuracy function to use to get training accuracy, if desired
        :kwargs - (Optional) specific hyperparameters for optimizer used. left unenumerated for cleanliness, since these vary optimizer to optimizer.
    """
    def __init__(self, 
                 X: ArrayType,
                 y: ArrayType,
                 P_S: int = None, 
                 optimizer_mode: str = "ADMM",
                 loss_type: str = 'mse',
                 datatype_backend: str = 'numpy',
                 standardize_data: bool = False,
                 acc_func: EvalFunction = classifcation_accuracy,
                 verbose_initialization: bool = False,
                 device: str = "cpu",
                 **kwargs,
                 ):
        
        if verbose_initialization: print("###### Initializing CReLU MLP! ########")

        # validate input training data
        assert len(X.shape) == 2, "X must be 2 dimensional with shape (n,d)"
        if len(y.shape) == 1: y = y[:,None]
        assert len(y.shape) == 2, "Y must have shape of (n,) or (n,1)"
        self.X, self.y = X, y 
        self.num_features = X.shape[1]

        # auto-set P_S value by using sqrt of feature dimension. TODO: set better heuristic for this
        if P_S is None: 
            P_S = int(mnp.sqrt(X.shape[1]))
            if verbose_initialization: print(f"\tAuto-setting P_S to sqrt of number of features: P_S={P_S}")

        # decide on optimizer function
        self.optimizer_mode = optimizer_mode
        if self.optimizer_mode in ["ADMM", "ADMM-RBCD"]:
            self.optimizer = admm_optimizer
        elif self.optimizer_mode == "CVXPY":
            self.optimizer = cvxpy_optimizer
        else:
            raise NotImplementedError(f"optimizer_mode must be one of {OPTIM_MODES}")
        if verbose_initialization: print(f"\tSelected mode: {optimizer_mode}. Using optimizer: {self.optimizer}")

        # datatype backend to use (numpy, torch, or jax)
        assert datatype_backend in DATATYPE_BACKENDS, f"Parameter \"datatype_backend\" must be one of {DATATYPE_BACKENDS}."
        self.datatype_backend = datatype_backend

        # ensure cvxpy only gets numpy backend
        if self.optimizer_mode == "CVXPY" and datatype_backend != "numpy":
            if verbose_initialization: print("\tWARNING: CVXPY requires numpy input; defaulting to numpy backend.")
            self.datatype_backend = "numpy"
        else:
            if verbose_initialization: print(f"\tUsing backend: {self.datatype_backend}.")
        
         # acceleration setup
        if device is None:
            self.device = mnp.get_device(self.datatype_backend)
        else:
            self.device = device

        # put data on correct backend
        X = convert_backend_type(X, self.datatype_backend, device=self.device)
        y = convert_backend_type(y, self.datatype_backend, device=self.device)
        
        # gather all hyperpameters to give to optimizer (We want this abstracted away from the user if possible, in the end state)
        self.parms = ADMM_Params(mode=self.optimizer_mode, 
                                 datatype_backend=self.datatype_backend, 
                                 device=self.device, 
                                 P_S=P_S, 
                                 loss_type=loss_type, 
                                 num_features=self.num_features, 
                                 **kwargs)

        # preprocessing parameters
        self.standardize_data = standardize_data
        self.bias = self.parms.bias

        # assessment parameters
        self.loss_func = cross_entropy_loss if loss_type == "ce" else squared_loss
        self.acc_func = acc_func

        # optimal weights of C-ReLU
        self.v = None
        self.w = None
        
        # optimal weights of NC-ReLU
        self.u = None
        self.alpha = None

        # optimization metrics for keeping track of performance
        self.training_metrics = {}

        # flag to ensure predictions only enabled after optimization called
        self.optimized = False

        if verbose_initialization: print("Initialization complete, ready for training with .optimize().")

    """
    Optimize cvx neural network with initialized optimization parameters
        :param max_iter (optional) - max iterations for ADMM algorithm
        :param verbose (optional) - true to print live optimiation progress
        :param X_val (optional) - optionally provide validation data to get val accuracy during each iteration
        :param y_val (optional) - labels associated with validation data
    """
    def optimize(self,
                 max_iter: int = 100, 
                 verbose: bool = False,
                 X_val: ArrayType = None,
                 y_val: ArrayType = None,):

        # preprocess training data
        self.X, self.y = self._preprocess_data(self.X, y=self.y)

        # if provided, also preprocess validation data
        if X_val is not None and y_val is not None:
            val_data = self._preprocess_data(X_val, y=y_val)
        else:
            val_data = None

        # solve optimization problem
        t_start = perf_counter()
        self.v, self.w, self.training_metrics = self.optimizer(self.parms, 
                                                      self.X, 
                                                      self.y, 
                                                      loss_func = self.loss_func, 
                                                      acc_func = self.acc_func, 
                                                      max_iter = max_iter, 
                                                      verbose = verbose,
                                                      val_data = val_data)
        self.training_metrics["solve_time"] = perf_counter() - t_start

        # recover u1.... u_ms and alpha1 ... alpha_ms as Optimal Weights of NC-ReLU Problem
        self.u, self.alpha = optimal_weights_transform(self.v, self.w, self.parms.P_S, self.parms.d, verbose=verbose)

        self.optimized = True

        if verbose: print("\nOptimization complete! Ready for application with .predict().\n")

        return self.training_metrics

    """
    Predict classes given new data X
        :param X - Evaluation data (* x num_features)
    """
    def predict(self, 
                X: ArrayType) -> ArrayType:

        assert self.optimized is True, "Must call .optimize() before applying predictions."
        assert len(X.shape) == 2, "X must be 2 dimensional array"
        assert X.shape[1] == self.num_features, f"X must have same feature size as trained data (d={self.num_features})"

        X = self._preprocess_data(X)

        # prediction using weights for equivalent nonconvex problem
        y_hat = mnp.relu(X @ self.u) @ self.alpha

        return y_hat
    
    """
    Get training metrics from solver
    """
    def get_training_metrics(self) -> dict:
        assert self.optimized is True, "Must call .optimize() to solve problem first."
        return self.training_metrics
    
    """
    Preprocess the data X (and optionally labels y) based on desired inputs (standardization, bias, etc)
    """
    def _preprocess_data(self, 
                         X: ArrayType,
                         y: ArrayType = None) -> ArrayType:
        
        # standardize data if desired
        if self.standardize_data:
            # standardize with data as np array
            X = convert_backend_type(X, "numpy", device=self.device)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # assert correct backend
        X = convert_backend_type(X, self.datatype_backend, device=self.device)

        # add bias term to data if desired
        if self.bias:
            X = mnp.hstack([X, mnp.ones((X.shape[0],1), backend_type=self.datatype_backend, device=self.device)])

        # convert labels if provided
        if y is not None:
            if len(y.shape) == 1: y = y[:,None]
            assert len(y.shape) == 2, "Y must have shape of (n,) or (n,1)"
            assert y.shape[0] == X.shape[0], "y must have n labels" 
            y = convert_backend_type(y, self.datatype_backend, device=self.device)
            return X, y
        else:
            return X