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
                self.admm_cg_solve_params = {
                    'cg_max_iters': 3,
                    'cg_eps': 1e-2,
                    'pcg': False,
                }
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
linear system solvers for ADMM
"""
class linear_sys:

    def __init__(self, OPS, rho, cg_params, solver_type=None, backend_type='numpy'):
        
        # Extract solver type
        solver_types = ['cg', 'cholesky']
        if solver_type.lower() not in solver_types:
            solver_type = 'cg'

        n = OPS.n
        d = OPS.d
        P_S = OPS.P_S
        
        # Cholesky 
        if solver_type == 'cholesky' or cg_params['pcg']:
            A = mnp.eye(2 * d * P_S, backend_type=backend_type)
            for i in range(P_S):
                for j in range(P_S):
                    # perform multiplication 
                    FiFj = OPS.F(i % P_S).T @ OPS.F(j % P_S) / rho
                    # assign to four quadrants
                    A[i*d:(i+1)*d, j*d:(j+1)*d] += FiFj
                    A[(i+P_S)*d:(i+P_S+1)*d, (j)*d:(j+1)*d] += - FiFj
                    A[(i)*d:(i+1)*d, (j+P_S)*d:(j+P_S+1)*d] += - FiFj
                    A[(i+P_S)*d:(i+P_S+1)*d, (j+P_S)*d:(j+P_S+1)*d] += FiFj
            for i in range(2):
                for j in range(P_S):
                    lower_ind = d * j + i * d * P_S
                    upper_ind = d * (j+1) + i * d * P_S
                    A[lower_ind:upper_ind, lower_ind:upper_ind] += OPS.G(j).T @ OPS.G(j)
            
            self.L = mnp.cholesky(A)

            if cg_params['pcg']:

                # Nys preconditiioner TODO            

                # Diagonal precond
                inds = mnp.arange(0, A.shape[0], backend_type=backend_type)
                diags = A[inds, inds]
                diags_tensor = vec_to_tensor(diags, d, P_S)
                self.M = lambda u : u / diags_tensor

                # No precond
                # self.M = lambda u : u
        
        self.n = n
        self.d = d
        self.rho = rho
        self.P_S = P_S
        self.OPS = OPS
        self.solver_type = solver_type
        self.backend_type = backend_type
        self.cg_params = cg_params
    
    def solve(self, b):
        u = None
        if self.solver_type == 'cg':
            eps = self.cg_params['cg_eps']
            max_cg_iter = self.cg_params['cg_max_iters']
            u = mnp.zeros((2, self.d, self.P_S), backend_type=self.backend_type)
            r = b.copy()
            nrm = eps * mnp.sqrt(mnp.sum(b ** 2))
            if self.cg_params['pcg']:
                z = self.M(r)
                p = z.copy()
                rho = r.flatten() @ z.flatten()
                rho_prev = rho.copy()
                for k in range(max_cg_iter):
                    if mnp.sqrt(rho) <= nrm:# or np.linalg.norm(r) <= nrm:
                        break
                    
                    w = p.copy()
                    w += 1/self.rho * self.OPS.F_multop(self.OPS.F_multop(p), transpose=True)
                    w += self.OPS.G_multop(self.OPS.G_multop(p), transpose=True)
                    alpha = rho / (p.flatten() @ w.flatten())
                    u = u + alpha * p
                    r = r - alpha * w
                    z = self.M(r)
                    rho_prev = rho.copy()
                    rho = z.flatten() @ r.flatten()
                    p = z + p * (rho / rho_prev)
            else:
                rho = r.flatten() @ r.flatten()
                rho_prev = rho + 0.0
                for k in range(max_cg_iter):
                    if mnp.sqrt(rho) <= nrm:
                        break
                    
                    if k == 0:
                        p = r.copy()
                    else:
                        p = r + (rho / rho_prev) * p
                    
                    w = p.copy()
                    w += 1/self.rho * self.OPS.F_multop(self.OPS.F_multop(p), transpose=True)
                    w += self.OPS.G_multop(self.OPS.G_multop(p), transpose=True)
                    alpha = rho / (p.flatten() @ w.flatten())
                    u = u + alpha * p
                    r = r - alpha * w
                    rho_prev = rho.copy()
                    rho = r.flatten() @ r.flatten()
        elif self.solver_type == 'cholesky':
            b = tensor_to_vec(b)
            bhat = mnp.solve_triangular(self.L, b, lower=True)
            u = mnp.solve_triangular(self.L.T, bhat, lower=False)
            u = vec_to_tensor(u, self.d, self.P_S)
        return u

"""
class to do F and G multiplicative operations a bit more memory efficiently
"""
class FG_Operators():

    def __init__(self, d_diags, X, backend_type='numpy'):
        n, P_S = d_diags.shape
        n, d = X.shape
        
        self.P_S = P_S
        self.n = n
        self.d = d
        self.d_diags = d_diags
        self.X = X
        self.mem_save = False
        self.backend_type = backend_type
        
        self.f_diag = mnp.vstack((1.0 * d_diags[None, ...], 
                                      -1.0 * d_diags[None, ...]))

    # get matrix F_i
    def F(self, i):
        return self.d_diags[:,i, None] * self.X
    
    # get matrix G_i
    def G(self, i):
        return (2 * self.d_diags[:, i, None] - 1) * self.X
    
    # replace linop F * vec
    def F_multop(self, vec, transpose=False):

        if self.mem_save:
            # @Zach's implimentation
            if transpose:
                vec = vec.squeeze()
                assert vec.shape == (self.n,)
                out = mnp.zeros((2, self.d, self.P_S))
                for i in range(self.P_S):
                    out[0,:,i] = self.F(i).T @ vec
                    out[1,:,i] -= self.F(i).T @ vec
            else:
                assert vec.shape == (2, self.d, self.P_S)
                out = mnp.zeros((self.n,), backend_type=self.backend_type)
                for i in range(self.P_S):
                    out += self.F(i) @ (vec[0,:,i] - vec[1,:,i])
        else:
            # @Daniel's implimentation
            if transpose:
                diags_to_vec = self.f_diag * vec.squeeze()[None, :, None]
                out = mnp.sum(diags_to_vec[:, :, None, :] * self.X[None, :, :, None], axis=1)
            else:
                vec_times_X = self.X @ vec
                out = mnp.sum(vec_times_X * self.f_diag, axis=(0, -1))

        return out
    
    # replace linop G * vec
    def G_multop(self, vec, transpose=False):
        
        if self.mem_save:
            # @Zach's implimentation
            out = mnp.zeros((2, self.d if transpose else self.n, self.P_S), backend_type=self.backend_type)

            for i in range(self.P_S):
                for j in range(2):
                    out[j,:,i] = (self.G(i).T if transpose else self.G(i)) @ vec[j,:,i]
        else:
            # @Daniel's implimentation
            if transpose:
                diags_to_vec = (2 * self.d_diags[None, ...] - 1) * vec
                out = mnp.sum(diags_to_vec[:, :, None, :] * self.X[..., None], axis=1)
            else:
                out = (self.X @ vec) * (2 * self.d_diags[None, ...] - 1)

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

def hadamard(m, backend_type):
    """
    Computes mxm hadamard matrix
    """

    if m == 2:
        return mnp.array([[1, 1],
                         [1, -1]], backend_type=backend_type)
    else:
        Hm2 = hadamard(m//2, backend_type)
        row2 = mnp.hstack((Hm2, -Hm2))
        row1 = mnp.hstack((Hm2, Hm2))
        return mnp.vstack((row1, row2))