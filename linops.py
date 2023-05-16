import sigpy as sp
import numpy as np

# The sigpy library uses this linear operator abstraction, which is aweosme.
# Basically, when applying some linear function A to an input x, you can of course 
# use the matrix representation of A, but it is often not as efficient as exploting 
# the matrix A's structure. In our case, G, F are repetative, and have a structure 
# that is worth exploting.
class G_linop(sp.linop.Linop):

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        super().__init__((2, n, P_S), (2, d, P_S))

        # Build linop
        M = sp.linop.MatMul(self.ishape, X)
        W = sp.linop.Multiply(M.oshape, (2.0 * d_diags[None, ...] - 1))
        self.linop = W * M

    def _apply(self, input):
        return self.linop * input
    
    def _adjoint_linop(self):
        return self.linop.H
    
    def _normal_linop(self):
        return self.linop.H * self.linop
    
class F_linop(sp.linop.Linop):

    def __init__(self, d_diags, X):
        n, P_S = d_diags.shape
        n, d = X.shape
        super().__init__((n,), (2, d, P_S))

        # Build linop
        diag_tot = np.concatenate((1.0 * d_diags[None, ...], -1.0 * d_diags[None, ...]), axis=0)
        M = sp.linop.MatMul(self.ishape, X)
        W = sp.linop.Multiply(M.oshape, diag_tot)
        S = sp.linop.Sum(W.oshape, axes=(0, 2))

        self.linop = S * W * M

    def _apply(self, input):
        return self.linop * input
    
    def _adjoint_linop(self):
        return self.linop.H
    
    def _normal_linop(self):
        return self.linop.H * self.linop
    
