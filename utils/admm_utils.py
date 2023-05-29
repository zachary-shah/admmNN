"""
Various classes and functions for ADMM 
"""

from utils.typing_utils import get_backend_type
import utils.math_utils as mnp

"""
Constants
"""
OPTIM_MODES = ["CVXPY", "ADMM", "ADMM-RBCD"]
OPTIM_LOSSES = ["mse", "ce"]

"""
Wrapper class for the parameters (including hyperparameters) for ADMM optimization
Validates that correct parameters are passed into each optimizer
    :param rho - fixed penalty parameter (rho > 0)
    :param step - step size constant (step > 0)
    :param beta - augmented lagrangian constant (beta > 0)
    :param bias - True to include bias to weights in first layer
    :param seed - (Optional) random seed
TODO: finish descriptions, typing
"""
class ADMM_Params():
    def __init__(self, 
                 mode="ADMM",
                 datatype_backend="numpy",
                 device="cpu",
                 num_features=None,
                 loss_type=None,
                 bias=None,
                 P_S=None, 
                 rho=None,
                 step=None,
                 beta=None, 
                 admm_cg_solve=None,
                 alpha0=None,
                 RBCD_blocksize=None,
                 RBCD_thresh=None,
                 RBCD_thresh_decay=None,
                 gamma_ratio=None,
                 gamma_ratio_decay=None,
                 base_buffer_size=None,
                 rho_increment=None,
                 seed=None,
                 verbose=True,
                 ):
        
        def validate_param(param, param_name, default_value):
            if param is None:
                if verbose: print(f"  - Warning: solver param \"{param_name}\" not specified. Using default: {default_value}.")
                return default_value
            return param
        
        # get backend information
        self.datatype_backend = datatype_backend
        self.device = device

        # input validation
        assert num_features is not None, f"Must specify feature dimension"
        assert mode in OPTIM_MODES, f"Optimization Mode must be one of {OPTIM_MODES}, but got {mode}"
        self.mode = mode
        if verbose: print(f"***** Initializing {mode} Parameters ****")

        assert loss_type in OPTIM_LOSSES, f"Loss type must be one of {OPTIM_LOSSES}, but got {loss_type}"

        assert P_S is not None, f"Must specify number of hidden layers (P_S)."

        self.loss_type = validate_param(loss_type, "loss_type", "mse")
        self.bias = validate_param(bias, "bias", True)
        self.d = num_features + int(bias)
        self.P_S = P_S

        # add random seed if desired
        self.seed = seed

        # add default parameters for specific optimization method if desired
        if mode == "ADMM":
            self.beta = validate_param(beta, "beta", 0.001)
            self.rho = validate_param(rho, "rho", 0.01)
            self.step = validate_param(step, "step", 0.01)
            self.gamma_ratio = validate_param(gamma_ratio, "gamma_ratio", self.step / self.rho)

            # setup for Conjugate Gradient options to solve ADMM
            if admm_cg_solve is None:
                print("  - Defaulting to solving ADMM step with a linear system solve. Specify \"admm_cg_solve=True\" to solve with conjugate gradient.")
                admm_cg_solve = False
            elif admm_cg_solve:
                """
                @Daniel: for you to populate parameters you would like to specify for conjugate gradient solve (preconditioners, etc.)
                """
                self.admm_cg_solve_params = {}
            self.admm_cg_solve = admm_cg_solve

        elif mode == "ADMM-RBCD":
            self.beta = validate_param(beta, "beta", 0.0001)
            self.rho = validate_param(rho, "rho", 0.02)
            self.alpha0 = validate_param(alpha0, "alpha0", 3e-6)
            self.RBCD_blocksize = validate_param(RBCD_blocksize, "RBCD_blocksize", 3)
            self.RBCD_thresh= validate_param(RBCD_thresh, "RBCD_thresh", 0.7)
            self.RBCD_thresh_decay= validate_param(RBCD_thresh_decay, "RBCD_thresh_decay", 0.96)
            self.gamma_ratio= validate_param(gamma_ratio, "gamma_ratio", 0.2)
            self.gamma_ratio_decay= validate_param(gamma_ratio_decay, "gamma_ratio_decay", 0.99)
            self.base_buffer_size= validate_param(base_buffer_size, "base_buffer_size", 8)
            self.rho_increment= validate_param(rho_increment, "rho_increment", 0.0001)


"""
class to do F and G multiplicative operations a bit more memory efficiently
# TODO: add docs and typing
"""
class FG_Operators():

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        
        self.P_S = P_S
        self.n = n
        self.d = d
        self.d_diags = d_diags
        self.X = X
        self.backend_type = get_backend_type(X)

    # get matrix F_i
    def F(self, i):
        return self.d_diags[:,i, None] * self.X
    
    # get matrix G_i
    def G(self, i):
        return (2 * self.d_diags[:, i, None] - 1) * self.X
    
    # replace linop F * vec
    def F_multop(self, vec, transpose=False):

        if transpose:
            vec = vec.squeeze()
            assert vec.shape == (self.n,)
            out = mnp.zeros((2, self.d, self.P_S), backend_type=self.backend_type)
            for i in range(self.P_S):
                out[0,:,i] = self.F(i).T @ vec
                out[1,:,i] -= self.F(i).T @ vec
        else:
            assert vec.shape == (2, self.d, self.P_S)
            out = mnp.zeros((self.n,), backend_type=self.backend_type)
            for i in range(self.P_S):
                out += self.F(i) @ (vec[0,:,i] - vec[1,:,i])

        return out
    
    # replace linop G * vec
    def G_multop(self, vec, transpose=False):
        
        out = mnp.zeros((2, self.d if transpose else self.n, self.P_S), backend_type=self.backend_type)

        for i in range(self.P_S):
            for j in range(2):
                out[j,:,i] = (self.G(i).T if transpose else self.G(i)) @ vec[j,:,i]

        return out
    
"""
Function to sample diagonals of D_i matrices given training data X, and random vectors h
return: a n x P matrix, where each column i is the diagonal entries for D_i
TODO: add typing
"""
def get_hyperplane_cuts(X, P, seed=None):

    backend_type = get_backend_type(X)

    if seed is not None: mnp.seed(seed, backend_type=backend_type)

    n,d = X.shape

    d_diags = X @ mnp.randn((d, P), backend_type=backend_type) >= 0

    # make unique
    d_diags = mnp.unique(d_diags, axis=1)

    # add samples if needed
    while d_diags.shape[1] != P:
        d_diags = mnp.append(d_diags, X @ mnp.randn((d, P), backend_type=backend_type) >= 0, axis=1)
        d_diags = mnp.unique(d_diags, axis=1)
        d_diags = d_diags[:, :P]

    return d_diags.astype("float")

def tensor_to_vec(tensor):
    """
    Flatten a (2,d,P_S) tensor into a (2*d*P_S,) vector
    TODO: add documentation, typing
    """
    backend_type = get_backend_type(tensor)

    # hacky way to make empty array
    vec = mnp.zeros(0, backend_type=backend_type)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[2]):
            vec = mnp.append(vec, tensor[i, :, j])
    return vec

def vec_to_tensor(vec, d, P_S):
    """
    Convert a (2*d*P_S,) vector to the (2,d,P_S) tensor
    TODO: add documentation, typing
    """
    backend_type = get_backend_type(vec)

    tensor = mnp.zeros((2, d, P_S), backend_type=backend_type)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[2]):
            inds = mnp.arange(d * j, d * (j + 1), backend_type=backend_type) + i * d * P_S
            tensor[i, :, j] = vec[inds]
    return tensor


def proxl2(z, beta, gamma):
    """
    Proximal l2 for ADMM update step on (v,w).
    TODO: add documentation, typing
    """

    if len(list(z.shape)) == 1:  # One-dimensional
        if mnp.norm(z) == 0:
            return z
        else:
            return mnp.relu(1 - beta * gamma / mnp.norm(z)) * z
    elif len(list(z.shape)) == 2:  # Two-dimensional
        norms = mnp.norm(z, axis=0)
        mask = norms > 0
        res = mnp.zeros_like(z)
        res[:, mask] = mnp.relu(1 - beta * gamma / norms[mask]) * z[:, mask]
        return res
    else:
        raise('Wrong dimensions')
