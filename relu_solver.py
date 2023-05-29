"""
C-ReLU Optimizer Wrapper Class
"""

from time import perf_counter

from sklearn.preprocessing import StandardScaler

from optimizers import admm_optimizer, cvxpy_optimizer

from utils.relu_utils import squared_loss, cross_entropy_loss, classifcation_accuracy
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
                 **kwargs,
                 ):
        
        if verbose_initialization: print("###### Initializing CReLU MLP! ########")

        # validate input training data
        assert len(X.shape) == 2, "X must be 2 dimensional with shape (n,d)"
        if len(y.shape) == 1: y = y[:,None]
        assert len(y.shape) == 2, "Y must have shape of (n,) or (n,1)"
        self.X, self.y = X, y 
        self.num_features = X.shape[1]

        # auto-set P_S value by using sqrt of shape. TODO: set better heuristic for this
        if P_S is None: 
            P_S = int(mnp.sqrt(X.shape[0]))
            if verbose_initialization: print(f"\tAuto-setting P_S to {P_S}")

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
        self.device = mnp.get_device(self.datatype_backend)

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
        self.metrics = {}

        # flag to ensure predictions only enabled after optimization called
        self.optimized = False

        if verbose_initialization: print("Initialization complete, ready for training with .optimize().")

    """
    Optimize cvx neural network with initialized optimization parameters
        :param max_iter (optional) - max iterations for ADMM algorithm
        :param verbose (optional) - true to print live optimiation progress
    """
    def optimize(self,
                 max_iter: int = 100, 
                 verbose: bool = False) -> None:

        # preprocess training data
        self.X = self._preprocess_data(self.X)

        # solve optimization problem
        t_start = perf_counter()
        self.v, self.w, self.metrics = self.optimizer(self.parms, 
                                                      self.X, 
                                                      self.y, 
                                                      loss_func = self.loss_func, 
                                                      acc_func = self.acc_func, 
                                                      max_iter = max_iter, 
                                                      verbose = verbose)
        self.metrics["solve_time"] = perf_counter() - t_start

        # recover u1.... u_ms and alpha1 ... alpha_ms as Optimal Weights of NC-ReLU Problem
        self._optimal_weights_transform(verbose=verbose)

        self.optimized = True

        if verbose: print("\nOptimization complete! Ready for application with .predict().\n")

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
    Get metrics from solver
    """
    def get_metrics(self) -> dict:
        assert self.optimized is True, "Must call .optimize() to solve problem first."
        return self.metrics
    
    """
    Preprocess the data X based on desired inputs (standardization, bias, etc)
    """
    def _preprocess_data(self, 
                         X: ArrayType) -> ArrayType:
        
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
            X = mnp.hstack([X, mnp.ones((X.shape[0],1), backend_type=self.datatype_backend)])
        return X
        
    """
    Given optimal v^*, w^* of convex problem (Eq (2.1)), derive the optimal weights u^*, alpha^* of the non-convex probllem (Eq (2.1))
    Applies Theorem 1 of Pilanci, Ergen 2020
    """
    def _optimal_weights_transform(self, 
                                   verbose: bool = False) -> None:
        
        assert self.v is not None
        assert self.w is not None

        if self.v.shape == (self.parms.P_S, self.parms.d):
            self.v = self.v.T
        if self.w.shape == (self.parms.P_S, self.parms.d):
            self.w = self.w.T

        # ensure shapes are correct
        assert self.v.shape == (self.parms.d, self.parms.P_S), f"Expected weight v shape to be ({self.parms.d},{self.parms.P_S}), but got {self.v.shape}"
        assert self.w.shape == (self.parms.d, self.parms.P_S), f"Expected weight w shape to be ({self.parms.d},{self.parms.P_S}), but got {self.w.shape}"

        if verbose: 
            print(f"\nDoing weight transform: ")
            v_shp = self.v.cpu().numpy().shape if self.datatype_backend == "torch" else self.v.shape
            w_shp = self.w.cpu().numpy().shape if self.datatype_backend == "torch" else self.w.shape
            print(f"  starting v shape: {v_shp}")
            print(f"  starting w shape: {w_shp}")
            print(f"  P_S: {self.parms.P_S}")
            print(f"  d: {self.parms.d}")

        alpha1 = mnp.sqrt(mnp.norm(self.v, 2, axis=0))
        mask1 = alpha1 != 0
        u1 = self.v[:, mask1] / alpha1[mask1]
        alpha2 = -mnp.sqrt(mnp.norm(self.w, 2, axis=0))
        mask2 = alpha2 != 0
        u2 = -self.w[:, mask2] / alpha2[mask2]

        self.u = mnp.append(u1, u2, axis=1)
        self.alpha = mnp.append(alpha1[mask1], alpha2[mask2])

        if verbose: 
            u_shp = self.u.cpu().numpy().shape if self.datatype_backend == "torch" else self.u.shape
            a_shp = self.alpha.cpu().numpy().shape if self.datatype_backend == "torch" else self.alpha.shape
            print(f"  transfomred u shape: {u_shp}")
            print(f"  transformed alpha shape: {a_shp}")  