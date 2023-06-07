"""
Do all math functions with appropriate backend
"""

import numpy as np
import torch
import jax.numpy as jnp
import random
from typing import Union, Sequence, Tuple

# numpy does not have solve_triangular; use scipy instead
import scipy.linalg as scipy_LA 
import jax.scipy.linalg as jax_LA

from utils.typing_utils import ArrayType, ScalarTypes, get_backend_type, convert_backend_type, DATATYPE_BACKENDS, TORCH_DTYPE, NP_DTYPE, JAX_DTYPE

def get_device(backend_type: str):
    """
    Get correct device according to backend in use
    TODO: jax backend setup
    """
    if backend_type == "jax":
        print("WARNING: JAX only set up on CPU. @MIRIA: for you to set up jax device in backend.")
        return "cpu"
    elif backend_type == "torch":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: # else numpy
        return "cpu"

########## SCALAR MATH OPERATIONS #################

def relu(x: ArrayType) -> ArrayType:
    """
    Calculates the relu activation correctly, depending on backend type.

    Parameters
    ----------
    x : ArrayType
       Input array

    Returns
    -------
    y: ArrayType
       Array with same shape as x, but all negative components turned to 0.
    """

    # handle case of scalar input x
    if type(x) in ScalarTypes:
        return np.maximum(x, 0)

    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.maximum(x, 0)
    elif backend_type == "torch":
        return torch.nn.functional.relu(x)
    else: # else numpy
        return np.maximum(x, 0)
    
def maximum(x1: Union[ArrayType, float, int],
            x2: Union[ArrayType, float, int]) -> Union[ArrayType, float, int]:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    x1, x2 : ArrayType | scalar
        The arrays holding the elements to be compared. 
        If x1.shape != x2.shape, they must be broadcastable to a common shape 
        (which becomes the shape of the output).

    Returns
    -------
    y : ArrayType | scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """

    # handle case of scalar input x1 and/or x2
    if type(x1) not in ScalarTypes:
        backend_type = get_backend_type(x1)
    elif type(x2) not in ScalarTypes:
        backend_type = get_backend_type(x2)
    else: # else input was 2 scalars
        return np.maximum(x1, x2)

    if backend_type == "jax":
        return jnp.maximum(x1, x2)
    elif backend_type == "torch":
        return torch.maximum(x1, x2)
    else: # else numpy
        return np.maximum(x1, x2)

def min(x: Union[ArrayType, float, int], 
        axis=None) -> Union[ArrayType, float, int]:
    """
    Get minimum of an array along an axis.

    Parameters
    ----------
    x : ArrayType | scalar
        The data over which the minimum value desired to be found.

    Returns
    -------
    y : ArrayType | scalar
        Minimum values along axis
    """

    # handle case of scalar input x1 and/or x2
    if type(x) not in ScalarTypes:
        backend_type = get_backend_type(x)
    else: # else input was 2 scalars
        return x

    if backend_type == "jax":
        return jnp.min(x, axis=axis)
    elif backend_type == "torch":
        if axis is not None:
            return torch.min(x, dim=axis)
        else:
            return torch.min(x)
    else: # else numpy
        return np.min(x, axis=axis)

def max(x: Union[ArrayType, float, int], 
        axis=None) -> Union[ArrayType, float, int]:
    """
    Get maximum of an array along an axis.

    Parameters
    ----------
    x : ArrayType | scalar
        The data over which the maximum value desired to be found.

    Returns
    -------
    y : ArrayType | scalar
        Maximum values along axis
    """

    # handle case of scalar input x1 and/or x2
    if type(x) not in ScalarTypes:
        backend_type = get_backend_type(x)
    else: # else input was 2 scalars
        return x

    if backend_type == "jax":
        return jnp.max(x, axis=axis)
    elif backend_type == "torch":
        if axis is not None:
            return torch.max(x, dim=axis)
        else:
            return torch.max(x)
    else: # else numpy
        return np.max(x, axis=axis)
    
def abs(x: Union[ArrayType, float, int], 
        axis=None) -> Union[ArrayType, float, int]:
    """
    Get scalar-wise absolute value of array.

    Parameters
    ----------
    x : ArrayType | scalar
        The data over which the absolute value desired to be found.

    Returns
    -------
    y : ArrayType | scalar
        absolute values
    """

    # handle case of scalar input x1 and/or x2
    if type(x) not in ScalarTypes:
        backend_type = get_backend_type(x)
    else: # else input was 2 scalars
        return np.abs(x)

    if backend_type == "jax":
        return jnp.abs(x)
    elif backend_type == "torch":
        return torch.abs(x)
    else: # else numpy
        return np.abs(x)
      
def norm(x: ArrayType,
         ord: Union[int, str] = None, 
         axis: int = None,
         keepdims: bool = False) -> Union[float, ArrayType]: 
    
    """
    Calculates norm function with appropriate backend.

    Parameters
    ----------
    x : ArrayType
       If axis is None, x must be 1-D or 2-D, unless ord is None. 
       If both axis and ord are None, the 2-norm of x.ravel will be returned.
    ord  : int or str, optional
        Order of the norm (see table under Notes). inf means numpy's inf object. The default is None.
    axis : int, optional
       If axis is an integer, it specifies the axis of x along which to compute the vector norms. 
       If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of 
       these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a 
       matrix norm (when x is 2-D) is returned. 
    keepdims : boolean, optional
       If this is set to True, the axes which are normed over are left in the result as dimensions 
       with size one. With this option the result will broadcast correctly against the original x.

    Returns
    -------
    n : ArrayType
       Norm of the matrix or vector(s)
    """
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    elif backend_type == "torch":
        return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)
    else: # else numpy
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

def sqrt(x: Union[float, ArrayType]) -> Union[float, ArrayType]:
    """
    Calculates square root with appropriate backend

    Parameters
    ----------
    x : ArrayType
       The value(s) whose square-roots are required.

    Returns
    -------
    y : ArrayType
       An array of the same shape as x, containing the positive square-root of each element in x.
    """
    
    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.sqrt(x)
    elif backend_type == "torch":
        return torch.sqrt(x)
    else: # else numpy or scalar
        return np.sqrt(x)
        
def log(x: Union[float, ArrayType]) -> Union[float, ArrayType]:
    """
    Calculates natural logarithm with appropriate backend.

    Parameters
    ----------
    x : ArrayType
       The values whose log are required.

    Returns
    -------
    y : ArrayType
       The natural logarithm of x, element-wise. This is a scalar if x is a scalar.
    """
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.log(x)
    elif backend_type == "torch":
        return torch.log(x)
    else: # else numpy
        return np.log(x)

def exp(x: Union[float, ArrayType]) -> Union[float, ArrayType]:
    """
    Calculates exponential with appropriate backend.

    Parameters
    ----------
    x : ArrayType
       The values whose exponential are required.

    Returns
    -------
    y : ArrayType
       Output array, element-wise exponential of x. This is a scalar if x is a scalar.
    """
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.exp(x)
    elif backend_type == "torch":
        return torch.exp(x)
    else: # else numpy
        return np.exp(x)

def sign(x: ArrayType) -> ArrayType:
    """
    Returns an element-wise indication of the sign of a number.

    Parameters
    ----------
    x : ArrayType
       Input values.

    Returns
    -------
    y : ArrayType
       The sign of x. This is a scalar if x is a scalar.
    """

    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.sign(x)
    elif backend_type == "torch":
        return torch.sign(x)
    else: # numpy
        return np.sign(x)
    
def sum(x: ArrayType,
        axis: Union[int, Tuple[int]] = None) -> ArrayType:
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    x : ArrayType
       Input values.
    axis : None or int or tuple of ints, optional
       Axis or axes along which a sum is performed. The default, 
       axis=None, will sum all of the elements of the input array. 
       If axis is negative it counts from the last to the first axis.

    Returns
    -------
    y : ArrayType
       An array with the same shape as a, with the specified axis removed.
       If a is a 0-d array, or if axis is None, a scalar is returned. If
       an output array is specified, a reference to out is returned.
    """

    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.sum(x, axis=axis)
    elif backend_type == "torch":
        return torch.sum(x, dim=axis)
    else: # numpy
        return np.sum(x, axis=axis)

def mean(x: ArrayType,
         axis: Union[int, Tuple[int]] = None) -> ArrayType:
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    x : ArrayType
       Input values.
    axis : None or int or tuple of ints, optional
       Axis or axes along which the means are computed. 
       The default is to compute the mean of the flattened array.

    Returns
    -------
    y : ArrayType
       Returns a new array containing the mean values.
    """

    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.mean(x, axis=axis)
    elif backend_type == "torch":
        return torch.mean(x, dim=axis)
    else: # numpy
        return np.mean(x, axis=axis)

def round(x: ArrayType,
          decimals: int = 0) -> ArrayType:
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    x : ArrayType
       Input values.
    decimals : int, optional
       Number of decimal places to round to (default: 0). If decimals is
       negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    y : ArrayType
       An array of the same type as a, containing the rounded values. 
    """

    backend_type = get_backend_type(x)
    
    if backend_type == "jax":
        return jnp.around(x, decimals=decimals)
    elif backend_type == "torch":
        return torch.round(x, decimals=decimals)
    else: # numpy
        return np.around(x, decimals=decimals)

def inf(backend_type: str = "numpy"):
    """
    Given string backend_type, get the positive infinity value
    """

    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"

    if backend_type == "jax":
        return jnp.inf
    elif backend_type == "torch":
        return torch.inf
    else:
        return np.inf

def spacing(x: ScalarTypes):
    """
    Get distance to next largest number
    """

    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.spacing(x)
    elif backend_type == "torch":
        dev = x.device
        x = x.cpu().numpy()
        return torch.Tensor(np.spacing(x)).to(TORCH_DTYPE).to(dev)
    else:
        return np.spacing(x)

########## MAKE ARRAYS ############################

def array(object: list,
          dtype: Union[int, float] = None,
          backend_type: str = "numpy",
          device: torch.device = "cpu") -> ArrayType:
    """
    Make an array from a list on specified backend. 
    If using torch, will generate a torch.tensor type.

    Parameters
    ----------
    object : list
        array-like list to construct array or tensor
    dtype : type
        Type for array. Use abstract types like "int" or "float" rather than backend specific types like "numpy.float64".
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"]. Default to numpy
    device : str
        Specify device to load tensor to. default is cpu.
   
    Returns
    -------
    arr : ndarray
        An ArrayType with type determined by backend_type, constructed from object list
    """

    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"
    assert dtype in [None, int, float], f"Datatype must be generic. Recieved {dtype} but expected either {int} or {float}."
    
    if backend_type == "jax":
        return jnp.array(object, dtype=JAX_DTYPE)
    elif backend_type == "torch":
        return torch.tensor(object, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        return np.array(object, dtype=NP_DTYPE)

def zeros(shape: Union[int, Tuple[int]],
          backend_type: str,
          dtype: type = float,
          device: torch.device = "cpu",) -> ArrayType: 

    """
    Get array of zeros with desired shape in correct backend

    Parameters
    ----------
    x : array_like
        Array with shape desired
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].
    dtype : type
        Type for array. Use abstract types like "int" or "float" rather than backend specific types like "numpy.float64".

    Returns
    -------
    zeros_arr : ndarray
        An array with shape `shape` with all zero values.
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"
    assert dtype in [int, float], f"Datatype must be generic. Recieved {dtype} but expected either {int} or {float}."
    
    if backend_type == "jax":
        return jnp.zeros(shape, dtype=dtype)
    elif backend_type == "torch":
        return torch.zeros(shape, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        return np.zeros(shape, dtype=dtype)

def ones(shape: Union[int, Tuple[int]],
          backend_type: str,
          dtype: type = float,
          device: torch.device = "cpu",) -> ArrayType: 

    """
    Get array of ones with desired shape in correct backend

    Parameters
    ----------
    x : array_like
        Array with shape desired
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].
    dtype : type
        Type for array. Use abstract types like "int" or "float" rather than backend specific types like "numpy.float64".

    Returns
    -------
    ones_arr : ndarray
        An array with shape `shape` with all ones values.
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"
    assert dtype in [int, float], f"Datatype must be generic. Recieved {dtype} but expected either {int} or {float}."
    
    if backend_type == "jax":
        return jnp.ones(shape, dtype=JAX_DTYPE)
    elif backend_type == "torch":
        return torch.ones(shape, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        return np.ones(shape, dtype=NP_DTYPE)

def zeros_like(x: ArrayType) -> ArrayType: 

    """
    Get array of zeros like shape of x in correct backend

    Parameters
    ----------
    x : array_like
        Array with shape desired

    Returns
    -------
    zeros_arr : ndarray
        An array with shape `x.shape` with all zero values.
    """
    
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.zeros_like(x, dtype=JAX_DTYPE)
    elif backend_type == "torch":
        return torch.zeros_like(x, dtype=TORCH_DTYPE)
    else: # else numpy
        return np.zeros_like(x, dtype=NP_DTYPE)
    
def ones_like(x: ArrayType) -> ArrayType: 

    """
    Get array of ones like shape of x in correct backend

    Parameters
    ----------
    x : array_like
        Array with shape desired

    Returns
    -------
    ones_arr : ndarray
        An array with shape `x.shape` with all ones values.
    """
    
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.ones_like(x, dtype=JAX_DTYPE)
    elif backend_type == "torch":
        return torch.ones_like(x, dtype=TORCH_DTYPE)
    else: # else numpy
        return np.ones_like(x, dtype=NP_DTYPE)

def arange(start: Union[int,float],
           stop: Union[int,float],
           backend_type: str,
           device: torch.device = "cpu",
           step: Union[int,float] = 1) -> ArrayType: 

    """
    Executes *.arange but requires start and stop positions. defaults step to 1

    Parameters
    ----------
    start : int
        value to start at
    start : int
        value 1 greater than end value in array
    step : int
        step to take in between values of array's entries
    device : torch.device
      Device to load torch.Tensor onto 

    Returns
    -------
    arr : ndarray
        Range array desired
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"
    
    if backend_type == "jax":
        return jnp.arange(start, stop, step)
    elif backend_type == "torch":
        return torch.arange(start, stop, step).to(device)
    else: # else numpy
        return np.arange(start, stop, step)   

def eye(N: int,
        M: int = None,
        dtype: Union[int, float] = None,
        device: torch.device = "cpu",
        backend_type: str = "numpy") -> ArrayType: 

    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    dtype : data-type, optional
      Data-type of the returned array. Must be either int or float.
    device : torch.device
      Device to load torch.Tensor onto 
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].

    Returns
    -------
    I : ArrayType of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"
    
    if backend_type == "jax":
        return jnp.eye(N, M=M, dtype=JAX_DTYPE)
    elif backend_type == "torch":
        if M is not None:
            return torch.eye(n=N, m=M, dtype=TORCH_DTYPE, device=device)
        else:
            return torch.eye(n=N, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        if dtype is None: dtype = float
        return np.eye(N, M=M, dtype=NP_DTYPE)

def diag(diags: ArrayType) -> ArrayType: 

    """
    Return a 2-D matrix with given diagonal entries

    Parameters
    ----------
    diags : ArrayType
      Diagonal entries

    Returns
    -------
    D : ArrayType 
      Square diagonal matrix with diags as the diagonal entries
    """

    backend_type = get_backend_type(diags)
        
    if backend_type == "jax":
        return jnp.diag(diags)
    elif backend_type == "torch":
        return torch.diag(diags)
    else: # else numpy
        return np.diag(diags)

def diagonal(X: ArrayType) -> ArrayType: 

    """
    Return a 1-D matrix which is the diagonal of matrix X

    Parameters
    ----------
    X : ArrayType
        2-D matrix

    Returns
    -------
    diags : ArrayType 
        1-D diagonals of X
    """
    
    backend_type = get_backend_type(X)
    
    if backend_type == "jax":
        return jnp.diagonal(X)
    elif backend_type == "torch":
        return torch.diagonal(X)
    else: # else numpy
        return np.diagonal(X)


########## MODIFY ARRAYS #########################

def append(arr: ArrayType, 
           values: ArrayType, 
           axis: int = None):
    """
    Append 2 things togheter along designated axis

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.
    """
    assert type(arr) == type(values), f"Typematch error. Arr1 is type {type(arr)}, but Arr2 is type {type(values)}."
    
    backend_type = get_backend_type(arr)

    if backend_type == "jax":
        return jnp.append(arr, values, axis=axis)
    elif backend_type == "torch":
        if axis is not None:
            return torch.cat((arr, values), dim=axis)
        else:
            return torch.cat((arr, values))
    else: # else numpy
        return np.append(arr, values, axis=axis)

def hstack(tup: Sequence[ArrayType]) -> ArrayType:

    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    Parameters
    ----------
    tup : sequence of ArrayTypes
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : ArrayType
        The array formed by stacking the given arrays.
    """
    backend_type = get_backend_type(tup[0])

    if backend_type == "jax":
        return jnp.hstack(tup)
    elif backend_type == "torch":
        return torch.hstack(tup)
    else: # else numpy
        return np.hstack(tup)

def vstack(tup: Sequence[ArrayType]) -> ArrayType:

    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    Parameters
    ----------
    tup : sequence of ArrayTypes
        The arrays must have the same shape along all but the first axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : ArrayType
        The array formed by stacking the given arrays.
    """
    
    backend_type = get_backend_type(tup[0])

    if backend_type == "jax":
        return jnp.vstack(tup)
    elif backend_type == "torch":
        return torch.vstack(tup)
    else: # else numpy
        return np.vstack(tup)

def reshape(arr: ArrayType,
            newshape: Union[int, Tuple[int]],
            ) -> ArrayType:
    """
    Gives a new shape to an array without changing its data. 
    Always reshapes with order 'C' (C-like index order).

    Parameters
    ----------
    arr : ArrayType
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : ArrayType
        This will be a new view object if possible; otherwise, it will
        be a copy.
    """
    backend_type = get_backend_type(arr)

    if backend_type == "jax":
        return jnp.reshape(arr, newshape)
    elif backend_type == "torch":
        # torch only handles tuples
        if type(newshape) == int:
            newshape = (newshape,)
        return torch.reshape(arr, newshape)
    else: # else numpy
        return np.reshape(arr, newshape)

def copy(x: ArrayType) -> ArrayType:
    """
    Return a copied version of array x

    Parameters
    ----------
    x : ArrayType
        Array to be copied.

    Returns
    -------
    y : ArrayType
        Copied array.
    """
    backend_type = get_backend_type(x)

    if backend_type == "jax":
        return jnp.copy(x)
    elif backend_type == "torch":
        devce = x.device
        return x.detach().clone().to(devce)
    else:
        return np.copy(x)


########## RANDOM OPS #############################

def seed(seed: Union[int, None], 
         backend_type: str) -> None:
    
    """
    Seed appropriate random generator

    Parameters
    ----------
    seed : int
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"

    # seed random as well
    random.seed(seed)

    if backend_type == "jax":
        jnp.random.seed(seed)
    elif backend_type == "torch":
        torch.manual_seed(seed)
    else: # else numpy
        np.random.seed(seed)

    return

def unique(ar: ArrayType, 
           axis: Union[int, None] = None) -> ArrayType:
    
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are two optional
    outputs in addition to the unique elements:

    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : ArrayType
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.

    Returns
    -------
    unique : ArrayType
        The sorted unique values.
    """

    backend_type = get_backend_type(ar)

    if backend_type == "jax":
        return jnp.unique(ar, axis=axis)
    elif backend_type == "torch":
        return torch.unique(ar, sorted=True, dim=axis)
    else: # else numpy
        return np.unique(ar, axis=axis)
    
def randn(size: Sequence[int],
          backend_type: str,
          device: str = "cpu") -> ArrayType:
    
    """
    Get array or tensor of random iid numbers sampled from normal distribution

    Parameters
    ----------
    size : sequence of integers
        Describes size desired for random generated array
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].

    Returns
    -------
    arr : ArrayType
        The random values desired.
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"

    if backend_type == "jax":
        y = np.random.randn(*size)
        return convert_backend_type(y, target_backend="jax").astype(JAX_DTYPE)
    elif backend_type == "torch":
        return torch.randn(*size, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        return np.random.randn(*size).astype(NP_DTYPE)
    
def rand(size: Sequence[int],
          backend_type: str,
          device: str = "cpu") -> ArrayType:
    
    """
    Get array or tensor of randoms numbers sampled from unif [0,1] distribution

    Parameters
    ----------
    size : sequence of integers
        Describes size desired for random generated array
    backend_type : str
        Backend type to create array in. Must be one of ["numpy", "jax", "torch"].

    Returns
    -------
    arr : ArrayType
        The random values desired.
    """
    
    assert backend_type in DATATYPE_BACKENDS, f"Input backend type {backend_type} incorrect; must be one of {DATATYPE_BACKENDS}"

    if backend_type == "jax":
        y = np.random.rand(*size)
        return convert_backend_type(y, target_backend="jax").astype(JAX_DTYPE)
    elif backend_type == "torch":
        return torch.rand(*size, dtype=TORCH_DTYPE, device=device)
    else: # else numpy
        return np.random.rand(*size).astype(NP_DTYPE)

def random_choice(a: Union[ArrayType, int],
                  size: Union[int, Tuple[int]] = None,
                  replace: bool = False,
                  p: ArrayType = None,
                  backend_type: str = None) -> ArrayType:
    """
    Generates a random sample from a given 1-D array.
    If backend is torch, returns numpy array, since numpy can be used for torch indexing
    If backend is jax, returns a jax.numpy array
    If backend is numpy, returns a numpy array

    Parameters
    ----------
    a : 1-D ArrayType or int
        If an ndarray, a random sample is generated from its elements. 
        If an int, the random sample is generated as if it were np.arange(a)
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), 
        then m * n * k samples are drawn. Default is None, 
        in which case a single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement. Default is True, 
        meaning that a value of a can be selected multiple times.
    p : 1-D array-like, optional
        The probabilities associated with each entry in a. If not given,
          the sample assumes a uniform distribution over all entries in a.

    Returns
    -------
    samples : ArrayType
        The generated random samples
    """

    assert backend_type is not None, "Must supply backend type for random_choice()."

    # just do random choice with numpy 
    i = np.random.choice(a, size=size, replace=replace, p=p)
    
    # if jax, then convert to jax indices
    if backend_type == "jax":
        return convert_backend_type(i, "jax").astype(int)
    # numpy and torch both use np array indexing
    else:
        return i

############ LINEAR ALGEBRA ####################

def cholesky(A: ArrayType) -> ArrayType:
    
    """
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

    Parameters
    ----------
    A : ArrayType
        Hermitian (symmetric if all elements are real), positive-definite input matrix.

    Returns
    -------
    L : ArrayType
        Lower-triangular Cholesky factor of a. Returns a matrix object if a is a matrix object.
    """

    backend_type = get_backend_type(A)
    
    if backend_type == "jax":
        return jnp.linalg.cholesky(A)
    elif backend_type == "torch":
        return torch.linalg.cholesky(A)
    else: # numpy
        return np.linalg.cholesky(A)

def dot(a: ArrayType,
        b: ArrayType) -> ArrayType:
    
    """
    Computes the dot product of 2 arrays.

    Parameters
    ----------
    a : ArrayType
        First argument array.
    b : ArrayType
        Second argument array.

    Returns
    -------
    output : ArrayType
        Returns the dot product of a and b. If a and b are both scalars 
        or both 1-D arrays then a scalar is returned; otherwise an 
        array is returned. If out is given, then it is returned.

    Note: For tensors, a and b must be 1-D arrays.

    """

    backend_type = get_backend_type(a)
    
    if backend_type == "jax":
        return jnp.dot(a, b)
    elif backend_type == "torch":
        # convert 2D to 1D if possible
        if len(a.shape) == 2 and len(a[0]) == 1:
            a = torch.reshape(a, (-1,))
        if len(b.shape) == 2 and len(b[0]) == 1:
            b = torch.reshape(b, (-1,))
        assert len(a.shape) == 1 and len(b.shape) == 1

        return torch.dot(a,b)
    
    else: # else numpy
        return np.dot(a,b)

def solve_triangular(a: ArrayType,
                     b: ArrayType,
                     lower: bool = False,
                     check_finite: bool = True) -> ArrayType:
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `a x = b`
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `a x = b`.  Shape of return matches `b`.
    """

    backend_type = get_backend_type(a)
    
    if backend_type == "jax":
        return jax_LA.solve_triangular(a, b, lower=lower, check_finite=check_finite)
    elif backend_type == "torch":
        if len(b.shape) == 1:
            return torch.linalg.solve_triangular(a, b[:,None], upper = not lower).reshape(-1)
        else:
            return torch.linalg.solve_triangular(a, b, upper = not lower)
    else: # numpy
        return scipy_LA.solve_triangular(a, b, lower=lower, check_finite=check_finite)

def inverse(A: ArrayType) -> ArrayType:
    """
    Computes the matrix inverse

    Parameters
    ----------
    A : ArrayType
        Square Matrix

    Returns
    -------
    A_inv : ArrayType
        Inverse of A
    """

    backend_type = get_backend_type(A)
    
    if backend_type == "jax":
        return jnp.linalg.inv(A)
    elif backend_type == "torch":
        return torch.linalg.inv(A)
    else: # numpy
        return np.linalg.inv(A)
    
def svd(A: ArrayType) -> ArrayType:
    """
    Computes the SVD of A

    Parameters
    ----------
    A : ArrayType
        Square Matrix

    Returns
    -------
    U : ArrayType
        Orthogonal matrix 'U'
    s : ArrayType
        singular values
    Vt : ArrayType
        Orthogonal matrix 'V' transpose
    """

    backend_type = get_backend_type(A)
    
    if backend_type == "jax":
        return jax_LA.svd(A, full_matrices=False, check_finite=False)
    elif backend_type == "torch":
        return torch.svd(A)
    else: # numpy
        return scipy_LA.svd(A, full_matrices=False, check_finite=False)

def qr(A: ArrayType) -> ArrayType:

    """
    Computes the QR deomposition of A

    Parameters
    ----------
    A : ArrayType
        Square Matrix

    Returns
    -------
    Q : ArrayType
        Orthogonal matrix
    R : ArrayType
        Triangular matrix
    """

    backend_type = get_backend_type(A)
    
    if backend_type == "jax":
        return jax_LA.qr(A, mode='economic')
    elif backend_type == "torch":
        return torch.linalg.qr(A, mode='reduced')
    else: # numpy
        return scipy_LA.qr(A, mode='economic')
    
def eigh(A: ArrayType,
         eigvals_only: bool = False) -> Union[Tuple[ArrayType,ArrayType],ArrayType]:

    """
    Get eigenvalues (and optionally also eigenvalues) of matrix

    Parameters
    ----------
    A : ArrayType
        Square Matrix
    eigvals_only : bool, default = False
        True to also return eigenvectors

    Returns
    -------
    w : (N,) ArrayType
        eigenvalues
    v : (M,N)
        eigenvectors (if eigvals_only == False)
    """

    backend_type = get_backend_type(A)
    
    if backend_type == "jax":
        return jax_LA.eigh(A, eigvals_only=eigvals_only)
    elif backend_type == "torch":
        if eigvals_only:
            return torch.linalg.eigvalsh(A)
        else:
            return torch.linalg.eigh(A)
    else: # numpy
        return scipy_LA.eigh(A, eigvals_only=eigvals_only)
    