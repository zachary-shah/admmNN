"""
Various classes and functions for ADMM 
"""

from utils.typing_utils import get_backend_type, as_default_datatype, EvalFunction, ArrayType
import utils.math_utils as mnp
from utils.cg_utils import nystrom_sketch

"""
Constants
"""
OPTIM_MODES = ["CVXPY", "ADMM", "ADMM-RBCD"]
OPTIM_LOSSES = ["mse", "ce"]
CG_PRECONDITIONERS = ['jacobi', 'sketch', 'nystrom']
ADMM_SOLVER_TYPES = ['cg', 'cholesky']

class ADMM_Params():
    """
    Wrapper class for the parameters (including hyperparameters) for ADMM optimization
    Validates that correct parameters are passed into each optimizer

    Parameters
    ----------
    mode : str = "ADMM"
        optimization mode. must be one of OPTIM_MODES
    datatype_backend : str = "numpy"
        compute backend. must be one of ["jax", "numpy", "torch"].
    device : str = "cpu"
        device to run on (cpu, cuda, etc)
    num_features : int
        number of features in training data X (without bias)
    loss_type : str
        type of loss used; either "mse" or "ce"
    memory_save : bool
        true to default to saving memory over being time-efficient in matrix computations (for F/G operator computations)
    bias : bool
        true to add bias term to training data
    P_S : int
        number of hyperplane samples
    rho : float
        fixed penalty parameter (rho > 0)
    step : float
        step size constant (step > 0)
    beta : float
        augmented lagrangian constant (beta > 0)
    alpha0 : float
        starting alpha for line search of RBCD primal update
    RBCD_blocksize : int
        how many samples in RBCD block (1 is blocksize of 1, like in paper)
    RBCD_thresh : float
        threshold on mean costs for ending RBCD update step
    RBCD_thresh_decay : float
        multiplicative decrease in RBCD threshold iteration to iteration
    gamma_ratio : float
        ratio of gamma to beta for dual update
    gamma_ratio_decay : float
        multiplicative decrease in gamma ratio iteration to iteration
    base_buffer_size : int
        how large of buffer to keep for distance costs during RBCD update
    rho_increment : float
        increase in rho iteration to iteration
    seed : int
        provide seed for repeatable results
    verbose : bool
        true to print information about parameters selected
    admm_solve_type : str
        solver type to use for mode="ADMM". default is "cholesky", but can use "cg" (conjugate gradient) instead
    cg_max_iters : int
        max iterations for cg solve, if selected
    cg_eps : float
        solve tolerance for cg, if selected
    cg_preconditioner : str
        preconditioner to use for cg. must be one of CG_PRECONDITIONERS
    """

    def __init__(self, 
                 mode: str = "ADMM",
                 datatype_backend: str = "numpy",
                 device: str = "cpu",
                 num_features: int = None,
                 loss_type: str = None,
                 memory_save: bool = None,
                 bias: bool = None,
                 P_S: int = None, 
                 rho: float = None,
                 step: float = None,
                 beta: float = None, 
                 alpha0: float = None,
                 RBCD_blocksize: int = None,
                 RBCD_thresh: float = None,
                 RBCD_thresh_decay: float = None,
                 gamma_ratio: float = None,
                 gamma_ratio_decay: float = None,
                 base_buffer_size: int = None,
                 rho_increment: float = None,
                 seed: int = None,
                 verbose: bool = False,
                 admm_solve_type: str = None,
                 cg_max_iters: int = None,
                 cg_eps: float = None,
                 cg_preconditioner: str = None,
                 ):
        
        def validate_param(param, param_name, default_value, value_list=None):

            if param is None:
                if verbose: print(f"  - Warning: solver param \"{param_name}\" not specified. Using default: {default_value}.")
                return default_value
            
            # if provided, require that a param is within the value list
            if value_list is not None: assert param in value_list, f"{param_name} must be one of {value_list}, but got {param}"

            return param
        
        # input validation
        self.mode = validate_param(mode, "mode", "ADMM", value_list=OPTIM_MODES)

        if verbose: print(f"***** Initializing {mode} Parameters ****")

        assert num_features is not None, f"Must specify feature dimension"
        assert P_S is not None, f"Must specify number of hidden layers (P_S)."

        # basic parameters
        self.loss_type = validate_param(loss_type, "loss_type", "mse", value_list=OPTIM_LOSSES)
        self.bias = validate_param(bias, "bias", True)
        self.memory_save = validate_param(memory_save, "memory_save", False)
        self.d = num_features + int(bias)
        self.P_S = P_S

        # backend information
        self.datatype_backend = datatype_backend
        self.device = device

        # add random seed if desired
        self.seed = seed

        # add default parameters for specific optimization method if desired
        if mode == "ADMM":
            self.beta = validate_param(beta, "beta", 0.001)
            self.rho = validate_param(rho, "rho", 0.01)
            self.step = validate_param(step, "step", 0.01)
            self.gamma_ratio = validate_param(gamma_ratio, "gamma_ratio", self.step / self.rho)

            self.admm_solve_type = validate_param(admm_solve_type, "admm_solve_type", 'cholesky', value_list=ADMM_SOLVER_TYPES)
            
            if verbose and admm_solve_type is None: print("  - Note: instead of defaulting to computing ADMM linear solve step with cholesky decomposition, specify admm_solve_type=\"cg\" to instead solve with conjugate gradient.")

            # setup for Conjugate Gradient options to solve ADMM
            self.admm_cg_solve_params = {}
            if self.admm_solve_type == 'cg':
                """
                Parameters for conjugate gradient solve (preconditioners, etc.)
                """
                self.admm_cg_solve_params = {
                    'cg_max_iters': validate_param(cg_max_iters, "cg_max_iters", 3),
                    'cg_eps': validate_param(cg_eps, "cg_eps", 1e-6),
                    'preconditioner': validate_param(cg_preconditioner, "cg_preconditioner", None, value_list=CG_PRECONDITIONERS),
                }

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

class FG_Operators():
    """
    Class to do F and G multiplicative operations a bit more memory efficiently

    Parameters
    ----------
    d_diags : ArrayType
        Diagonals of D_h matrices in vector form (shape P_S x n)
    X : ArrayType
        training data (n x d)
    rho : float
        fixed penalty parameter (rho > 0) in ADMM objective
    mem_save : bool = False
        False uses time-efficient computations; True uses memory-efficient computations

    """
    def __init__(self, d_diags: ArrayType,
                 X: ArrayType, 
                 rho: float = None, 
                 mem_save: bool = False):
        
        n, P_S = d_diags.shape
        n, d = X.shape
        
        self.P_S = P_S
        self.n = n
        self.d = d
        self.d_diags = d_diags
        self.X = X
        self.rho = rho
        self.mem_save = mem_save
        self.backend_type = get_backend_type(X)
        
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
                    if self.backend_type == "jax":
                        out = out.at[0,:,i].set(self.F(i).T @ vec)
                        out = out.at[1,:,i].set(- self.F(i).T @ vec)
                    else:
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
                    if self.backend_type == "jax":
                        out = out.at[j,:,i].set((self.G(i).T if transpose else self.G(i)) @ vec[j,:,i])
                    else:
                        out[j,:,i] = (self.G(i).T if transpose else self.G(i)) @ vec[j,:,i]
        else:
            # @Daniel's implimentation
            if transpose:
                diags_to_vec = (2 * self.d_diags[None, ...] - 1) * vec
                out = mnp.sum(diags_to_vec[:, :, None, :] * self.X[..., None], axis=1)
            else:
                out = (self.X @ vec) * (2 * self.d_diags[None, ...] - 1)

        return out
    
    # effectively does linop A * x = b in tensor format
    def A(self, vec):
        assert self.rho is not None, "Must supply rho in input to use b = A * p operator" 
        b = mnp.copy(vec)
        b += 1/self.rho * self.F_multop(self.F_multop(vec), transpose=True)
        b += self.G_multop(self.G_multop(vec), transpose=True)
        return b

class Linear_Sys():
    """
    Linear system solvers for ADMM. Initialization completes necessary precomputations:
    
    Cholesky, Nystrom or Sketch PCG: 
        - Requires A matrix formulation (expensive operation)
    Jacobi PCG:
        - Minimal precomputation of matrix diagonal
    CG:
        - No precomputation!

    Note: linear system is solved in tensor format: Equation A x = b has x, b with shape (2,d,P_S); 
    A either has abstract shape (operations tensorized with FG_Operators) or system is vectorized if A matrix
    requires explicit computatoin (like for Cholesky, Nystrom PCG)

    Parameters
    ----------
    OPS: FG_Operators
        object containing F and G multiplicative operations for matrix computation steps
    params: ADMM_Params
        parameters object from solver instantiation; includes all necessary solver and problem parameters
    verbose : bool (optional)
        true to print information about parameters selected
    """
    def __init__(self, 
                 OPS: FG_Operators,
                 params: ADMM_Params,
                 verbose: bool = False,
                ):

        n = OPS.n
        d = OPS.d
        P_S = OPS.P_S

        self.n = n
        self.d = d
        self.rho = params.rho
        self.P_S = P_S
        self.OPS = OPS

        if params.admm_solve_type == 'cg' and params.admm_cg_solve_params['preconditioner'] is not None:
            self.solver_type = 'pcg'
        else:
            self.solver_type = params.admm_solve_type

        self.backend_type = params.datatype_backend
        self.cg_params = params.admm_cg_solve_params
        self.verbose = verbose

        # Construct matrix A 
        if verbose: 
            print(f"Initializing linear system solver!\n\tSolver type = {self.solver_type}")
            if params.admm_solve_type == 'cg':
                print(f"\tCG params: {self.cg_params}")
        
        # construct matrix A unless using cg or jacobi pcg (these don't require matrix A computed)
        if self.solver_type == 'cholesky' or (self.solver_type == 'pcg' and self.cg_params['preconditioner'].lower() != "jacobi") :
            if verbose: print("Constructing A matrix...")

            A = mnp.eye(2 * d * P_S, backend_type=params.datatype_backend)
            for i in range(P_S):
                for j in range(P_S):
                    # perform multiplication 
                    FiFj = OPS.F(i % P_S).T @ OPS.F(j % P_S) / self.rho

                    # assign to four quadrants
                    if params.datatype_backend == "jax":
                        A = A.at[i*d:(i+1)*d, j*d:(j+1)*d].add(FiFj)
                        A = A.at[(i+P_S)*d:(i+P_S+1)*d, (j)*d:(j+1)*d].add(- FiFj)
                        A = A.at[(i)*d:(i+1)*d, (j+P_S)*d:(j+P_S+1)*d].add(- FiFj)
                        A = A.at[(i+P_S)*d:(i+P_S+1)*d, (j+P_S)*d:(j+P_S+1)*d].add(FiFj)
                    else:
                        A[i*d:(i+1)*d, j*d:(j+1)*d] += FiFj
                        A[(i+P_S)*d:(i+P_S+1)*d, (j)*d:(j+1)*d] += - FiFj
                        A[(i)*d:(i+1)*d, (j+P_S)*d:(j+P_S+1)*d] += - FiFj
                        A[(i+P_S)*d:(i+P_S+1)*d, (j+P_S)*d:(j+P_S+1)*d] += FiFj

            XTX = OPS.X.T @ OPS.X
            for i in range(2):
                for j in range(P_S):
                    lower_ind = d * j + i * d * P_S
                    upper_ind = d * (j+1) + i * d * P_S
                    # assign to four quadrants
                    if params.datatype_backend == "jax":
                        A = A.at[lower_ind:upper_ind, lower_ind:upper_ind].add(XTX)
                    else:                
                        A[lower_ind:upper_ind, lower_ind:upper_ind] += XTX
            
        # construct preconditioner
        if self.solver_type == 'pcg':

            if self.cg_params['preconditioner'].lower() == "jacobi":
                if verbose: print("  Using Jacobi diagonal preconditioner")

                # form only necessary parts of A to make preconditioner
                diags_tensor = mnp.ones((2, d, P_S), backend_type=params.datatype_backend)
                XiX_diag = mnp.diagonal(OPS.X.T @ OPS.X)

                for i in range(P_S):
                    if params.datatype_backend == "jax":
                        diags_tensor = diags_tensor.at[:,:,i].add(XiX_diag + mnp.diagonal(OPS.F(i).T @ OPS.F(i)))
                    else:                
                        elem = XiX_diag + mnp.diagonal(OPS.F(i).T @ OPS.F(i)) / self.rho
                        diags_tensor[0,:,i] += elem
                        diags_tensor[1,:,i] += elem

                self.M = lambda u : u / diags_tensor

            elif self.cg_params['preconditioner'].lower() == "nystrom":
                rank = 10 #TODO: set rank more intelligently
                if verbose: print(f"  Using Nystrom preconditioner with rank={rank} (TODO: parameterize / smart set rank)")

                # compute randomized nystrom approximation
                self.U, self.S = nystrom_sketch(A, rank=rank)

                # this is how to apply the approximation
                def nystrom_precond(u):
                    u = tensor_to_vec(u)
                    Utu = self.U.T @ u
                    u = (self.S[-1] + self.rho) * (self.U @ (Utu / (self.S + self.rho))) + u - self.U @ Utu
                    u = vec_to_tensor(u, self.d, self.P_S)
                    return u

                self.M = nystrom_precond
            
            elif self.cg_params['preconditioner'].lower() == "sketch":
                if verbose: print("  Using Sketch preconditioner")
                # Sketch preconditiioner TODO 
                raise NotImplementedError("Sketch preconditioner not yet implemented.")

            else:
                raise NotImplementedError(f"Unexpected preconditioner recieved. Must be one of {CG_PRECONDITIONERS}.")

        elif self.solver_type == 'cholesky':
            self.L = mnp.cholesky(A)
        elif self.solver_type == 'cg':
            pass
        else:
            raise NotImplementedError("Unexpected solver type received. Must be one of \'cg\' or \'cholesky\'.")
    

    def solve(self, b):

        """
        Solve the system A x = b in tensor or vectorized format
        """
        u = None

        # solve with pcg
        if self.solver_type == 'pcg':
            if self.verbose: print("\tBegininning pcg solve!")

            u = mnp.zeros_like(b)
            r = mnp.copy(b)
            z = self.M(r)
            
            p = mnp.copy(z)
            eps_nrm = self.cg_params['cg_eps'] * mnp.norm(b) 

            def iter_cond(k, rho):
                iter_flag = True
                if k > self.cg_params['cg_max_iters']:
                    if self.verbose: print(f"\tStopping cg due to max_cg_iters reached.")
                    iter_flag = False
                if mnp.norm(r) <= eps_nrm:
                    if self.verbose: print("\tStopping cg due to small residual norm.")
                    iter_flag = False
                return iter_flag 

            rho = mnp.sum(r * z)

            k = 1
            while iter_cond(k, r):

                # compute residual norm if printing metric
                if self.verbose: print(f"\t cg iter k = {k}, norm r = {mnp.norm(r)}, norm residual = {mnp.norm(b - self.OPS.A(u)) / mnp.norm(b)}, stopping norm = {eps_nrm}")

                # compute w = A @ p
                w = self.OPS.A(p)

                # perturbations
                alpha = rho / mnp.sum(w * p)
                u += alpha * p
                r -= alpha * w

                # apply preconditioner
                z = self.M(r)

                # updates
                rho_plus = mnp.sum(r * z)
                p = z + (rho_plus / rho) * p
                rho = rho_plus
                k += 1

        # solve unconditioned conjugate gradient
        elif self.solver_type == 'cg':
            if self.verbose: print("\tBegininning cg solve!")

            u = mnp.zeros_like(b)
            r, p = mnp.copy(b), mnp.copy(b)
            nrm = self.cg_params['cg_eps'] * mnp.norm(b)

            def iter_cond(k, rho):
                iter_flag = True
                if k > self.cg_params['cg_max_iters']:
                    if self.verbose: print(f"\tStopping cg due to max_cg_iters reached.")
                    iter_flag = False
                if mnp.sqrt(rho[k-1]) <= nrm:
                    if self.verbose: print("\tStopping cg due to small rho.")
                    iter_flag = False
                return iter_flag 

            rho = [mnp.sum(r * r)]

            k = 1
            while iter_cond(k, rho):

                # compute residual norm if printing metric
                if self.verbose: print(f"\t cg iter k = {k}, residual norm = {mnp.norm(b - self.OPS.A(u)) / mnp.norm(b)}")
                
                # base case
                if k > 1: p = r + (rho[k-1] / rho[k-2]) * p

                # compute w = A @ p
                w = self.OPS.A(p)

                # update as needed
                alpha = rho[k-1] / mnp.sum(p * w)
                u += alpha * p
                r -= alpha * w
                rho.append(mnp.sum(r * r))
                k += 1

        # cholesky triangular solves
        elif self.solver_type == 'cholesky':
            b = tensor_to_vec(b)
            bhat = mnp.solve_triangular(self.L, b, lower=True)
            u = mnp.solve_triangular(self.L.T, bhat, lower=False)
            u = vec_to_tensor(u, self.d, self.P_S)

        return u

def get_hyperplane_cuts(X: ArrayType, 
                        P: int, 
                        seed=None) -> ArrayType:
    """
    Function to sample diagonals of D_i matrices given training data X

    Parameters
    ----------
    X : ArrayType
        training data
    P : int
        number of unique hyperplane samples desired
    seed : int (optional)
        randomized seed for repeatable results

    Returns
    ----------
    d_diags: ArrayType
        n x P matrix, where each column i is the diagonal entries for D_i
    """

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
    
    # make default datatype
    return as_default_datatype(d_diags)

def tensor_to_vec(tensor: ArrayType) -> ArrayType:
    """
    Flatten a (2,d,P_S) tensor into a (2*d*P_S,) vector

    Parameters
    ----------
    tensor : ArrayType
        data in shape (2, d, P_S)

    Returns
    ----------
    vec: ArrayType
        data in shape (2 * d * P_S), with columns stacked sequentially by axis 0, then axis 2
    """

    backend_type = get_backend_type(tensor)

    # hacky way to make empty array
    vec = mnp.zeros(0, backend_type=backend_type)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[2]):
            vec = mnp.append(vec, tensor[i, :, j])
    return vec

def vec_to_tensor(vec: ArrayType, 
                  d: int, 
                  P_S: int) -> ArrayType:
    """
    Tensorize a (2*d*P_S) vector

    Parameters
    ----------
    vec: ArrayType
        data in shape (2 * d * P_S)
    d : int
        feature dimension
    P_S : int
        hyperplane samples 

    Returns
    ----------
    tensor : ArrayType
        data in shape (2, d, P_S)
    """
    backend_type = get_backend_type(vec)

    tensor = mnp.zeros((2, d, P_S), backend_type=backend_type)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[2]):
            inds = mnp.arange(d * j, d * (j + 1), backend_type=backend_type) + i * d * P_S
            if backend_type == "jax":
                tensor = tensor.at[i, :, j].set(vec[inds])
            else:
                tensor[i, :, j] = vec[inds]
    return tensor

def proxl2(z: ArrayType, 
           beta: float, 
           gamma: float):
    """
    Proximal l2 for ADMM update step on (v,w).
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
        if get_backend_type(z) == "jax":
            res = res.at[:, mask].set(mnp.relu(1 - beta * gamma / norms[mask]) * z[:, mask])
        else:
            res[:, mask] = mnp.relu(1 - beta * gamma / norms[mask]) * z[:, mask]
        return res
    else:
        raise('Wrong dimensions')
