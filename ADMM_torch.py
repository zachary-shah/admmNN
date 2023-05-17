# Performance measurement
from time import perf_counter
# Mathematics
import torch
import torch.nn.functional as F
import numpy as np
from math import ceil
# Helper functions
from utils import proxl2, generate_D
from postprocess import calculate_cost, evaluate, evaluate2, recover_weights


class ADMMTrainer:
    def __init__(self, X, y, P, beta=1e-4, iters=1000, rho=1, dmat=None, loss_type='mse',
                 vs=None, ws=None, X_test=None, y_test=None, alpha0=2e-4, RBCD_block_size=1,
                 RBCDthresh=1.3e-3, RBCDthresh_decay=.94, gamma_ratio=1/3, gamma_ratio_decay=.99):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        # Dataset
        self.X = torch.from_numpy(X).float().to(self.device)
        self.y = torch.from_numpy(y).float().to(self.device)
        self.X_test = torch.from_numpy(X_test).float().to(self.device)
        self.y_test = torch.from_numpy(y_test).float().to(self.device)
        (self.n, self.d), self.P = self.X.shape, P
        if dmat is not None:
            self.dmat = torch.from_numpy(dmat).float().to(self.device)
            self.emat = 2 * self.dmat - 1
            assert list(self.dmat.shape) == [self.n, self.P]
        else:
            self.dmat, self.emat = None, None
        print("Calculating GTG...")
        self.GTG = self.X.T @ self.X

        # Training settings
        self.beta, self.iters, self.rho = beta, iters, rho
        self.alpha0 = alpha0
        self.RBCDthresh, self.RBCDthresh_decay = RBCDthresh, RBCDthresh_decay
        self.RBCD_block_size = RBCD_block_size
        self.gamma_ratio, self.gamma_ratio_decay = gamma_ratio, gamma_ratio_decay
        self.loss_type = loss_type

        # Initialize weights
        self.u = torch.zeros([self.d, self.P * 2], device=self.device)
        self.v = torch.zeros([self.d, self.P], device=self.device)
        self.w = torch.zeros([self.d, self.P], device=self.device)
        self.s = torch.zeros([self.n, self.P], device=self.device)
        self.t = torch.zeros([self.n, self.P], device=self.device)
        self.lam1 = torch.zeros([self.d, self.P], device=self.device)
        self.lam2 = torch.zeros([self.d, self.P], device=self.device)
        self.nu1 = torch.zeros([self.n, self.P], device=self.device)
        self.nu2 = torch.zeros([self.n, self.P], device=self.device)
        self.vs, self.ws = vs, ws  # Ground truth weights (optional)

        # Initialize objectives
        self.costs = np.empty(self.iters + 1)
        self.costs2 = np.empty(self.iters + 1)
        self.dists = np.empty(self.iters + 1)
        self.accuracies = np.empty(self.iters + 1)

    def set_dmat_emat(self):
        if self.dmat is None:
            print('Generating D matrices...')
            dmat, self.n, self.d, self.P, v, w = generate_D(self.X.cpu().numpy(), self.P)  # dmat.shape is (n, P)
            self.dmat = torch.from_numpy(dmat).float().to(self.device)
            self.emat = 2 * self.dmat - 1  # n by P
        print('n d P:', self.n, self.d, self.P)

    def get_step_size(self, u, z, yhat, i, gradu, gradz, loss1):
        # This algorithm selects the step size for the RBCD subroutine.
        alpha = self.alpha0
        while True:  # Emulate a do-while loop
            du = -alpha * gradu
            dz = -alpha * gradz

            # Current prediction (via convex formulation)
            yhat_new = yhat + (self.dmat[:, i] * (self.X @ (du - dz))).sum(dim=1)
            if self.loss_type == 'mse':
                dloss = ((yhat_new - self.y) ** 2).sum() / 2 - loss1
            elif self.loss_type == 'ce':
                dloss = (-2 * self.y * yhat_new + torch.log(torch.exp(2 * yhat_new) + 1)).sum() - loss1

            ddist1 = ((u[:, i] + du - self.v[:, i] + self.lam1[:, i]) ** 2).sum() + \
                     ((z[:, i] + dz - self.w[:, i] + self.lam2[:, i]) ** 2).sum() - \
                     ((u[:, i] - self.v[:, i] + self.lam1[:, i]) ** 2).sum() + \
                     ((z[:, i] - self.w[:, i] + self.lam2[:, i]) ** 2).sum()
            ddist2 = ((self.emat[:, i] * (self.X @ (u[:, i] + du)) - self.s[:, i] + self.nu1[:, i]) ** 2).sum() + \
                     ((self.emat[:, i] * (self.X @ (z[:, i] + dz)) - self.t[:, i] + self.nu2[:, i]) ** 2).sum() - \
                     ((self.emat[:, i] * (self.X @ u[:, i]) - self.s[:, i] + self.nu1[:, i]) ** 2).sum() - \
                     ((self.emat[:, i] * (self.X @ z[:, i]) - self.t[:, i] + self.nu2[:, i]) ** 2).sum()
            dcost = dloss + (ddist1 + ddist2) * self.rho / 2

            # Armijo's rule
            if alpha <= 1e-8 or dcost <= -1e-3 * torch.sqrt((du ** 2).sum() + (dz ** 2).sum()):
                break
            alpha /= 2.5
            # Decaying basic step size
            self.alpha0 = np.maximum(1e-10, self.alpha0 / 1.5)

        return du, dz, yhat_new, dcost, alpha

    def RBCD_update(self, verbose=False):
        # Initialize optimization variables
        u, z = self.u[:, :self.P], self.u[:, self.P:]
        # Initialize matrices
        stil = self.X.T @ (self.emat * (self.s - self.nu1))
        ttil = self.X.T @ (self.emat * (self.t - self.nu2))
        # Initialize objective
        yhat = (self.dmat * (self.X @ (u - z))).sum(dim=1)  # .reshape(-1,1)

        # Terminate condition
        base_buffer_size = 8
        # Initialize circular buffer to be a finite large number
        dcosts = torch.ones(ceil(base_buffer_size * np.sqrt(self.P / self.RBCD_block_size))) * 1e8
        ptr, k = 0, 0  # k is current count of iterations
        while dcosts.mean() > self.RBCDthresh:
            k += 1
            i = np.random.choice(self.P, size=self.RBCD_block_size, replace=False)

            # Calculate training loss via u and z (without regularization) and get gradients
            if self.loss_type == 'mse':
                loss1 = ((yhat - self.y) ** 2).sum() / 2
                grad1 = self.X.T @ (self.dmat[:, i] * (yhat - self.y).unsqueeze(1))
            elif self.loss_type == 'ce':
                loss1 = (-2 * self.y * yhat + torch.log(torch.exp(2 * yhat) + 1)).sum()
                grad1 = self.X.T @ (self.dmat[:, i] * (-2 * self.y + 2 / (1 + torch.exp(-2 * yhat))).unsqueeze(1))
            else:
                raise ValueError("Unknown loss.")

            grad2u = u[:, i] - self.v[:, i] + self.lam1[:, i] + self.GTG @ u[:, i] - stil[:, i]
            grad2z = z[:, i] - self.w[:, i] + self.lam2[:, i] + self.GTG @ z[:, i] - ttil[:, i]
            gradu = grad1 + self.rho * grad2u
            gradz = -grad1 + self.rho * grad2z

            # Determine the step size
            du, dz, yhat, dcost, alpha = self.get_step_size(u, z, yhat, i, gradu, gradz, loss1)

            # Update u, z, and objective
            u[:, i] += du
            z[:, i] += dz
            dcosts[ptr] = -dcost
            # Update circular buffer
            ptr = (ptr + 1) % ceil(base_buffer_size * np.sqrt(self.P / self.RBCD_block_size))

            self.alpha0 *= 1.05
            if verbose and k % 20 == 0:
                print('Iteration', k, ', alpha:', alpha, ', delta:', dcosts.mean().item())
        self.u = torch.cat([u, z], dim=1)

    def calc_loss_acc(self, k, verbose=True):  # Calculate cost & accuracy
        # Get predictions using u and z
        yhat = (self.dmat * (self.X @ (self.u[:, :self.P] - self.u[:, self.P:]))).sum(dim=1)

        # Calculate convex objective (lvw)
        if self.loss_type == 'mse':
            cur_cost = ((yhat - self.y) ** 2).sum() / 2 + \
                       self.beta * ((torch.linalg.norm(self.v, dim=0)).sum() + (torch.linalg.norm(self.w, dim=0)).sum())
        elif self.loss_type == 'ce':
            cur_cost = (-2 * yhat * self.y + torch.log(torch.exp(2 * yhat) + 1)).sum() + \
                       self.beta * ((torch.linalg.norm(self.v, dim=0)).sum() + (torch.linalg.norm(self.w, dim=0)).sum())
        self.costs[k] = cur_cost.item()

        if self.vs is not None and self.ws is not None:  # If ground truth weights specified, then print distance.
            self.dists[k] = torch.sqrt(torch.linalg.norm(self.u[:, :self.P] - self.vs, ord='fro') ** 2 +
                                       torch.linalg.norm(self.u[:, self.P:] - self.ws, ord='fro') ** 2).item()
            if verbose:
                print('Iteration', k, ', Training cost:', self.costs[k], ', dists:', self.dists[k])
        else:  # If ground truth weights not specified, then only print cost.
            if verbose:
                print('Iteration', k, ', Training cost:', self.costs[k])

        # Recover weights and evaluate
        uu, alpha = recover_weights(self.v, self.w, use_torch=True, verbose=False)
        self.costs2[k] = calculate_cost(self.X, self.y, uu, alpha, beta=self.beta,
                                        loss_type=self.loss_type, use_torch=True, verbose=verbose)
        if self.X_test is not None and self.y_test is not None:
            accuracy, yhat = evaluate(self.X_test, self.y_test, uu, alpha, use_torch=True, verbose=verbose)
            # accuracy2, yhat2 = evaluate2(self.X_test, self.y_test, uu, alpha, verbose=verbose)
            self.accuracies[k] = accuracy

        return uu, alpha

    def ADMM_train_step(self, verbose=True, RBCD_verbose=True):
        ts = perf_counter()
        # First, update self.u (u and z) via RBCD
        self.RBCD_update(verbose=RBCD_verbose)
        self.RBCDthresh *= .96
        self.gamma_ratio *= .99
        self.rho += 0.0001
        # Then, update v and w
        self.v = proxl2(self.u[:, :self.P] + self.lam1, beta=self.beta, gamma=1 / self.rho, use_torch=True)
        self.w = proxl2(self.u[:, self.P:] + self.lam2, beta=self.beta, gamma=1 / self.rho, use_torch=True)
        # Next, update s and t
        Gu1 = self.emat * (self.X @ self.u[:, :self.P])
        Gu2 = self.emat * (self.X @ self.u[:, self.P:])
        self.s, self.t = F.relu(Gu1 + self.nu1), F.relu(Gu2 + self.nu2)
        # Finally, update the dual variables
        self.lam1 += self.gamma_ratio * (self.u[:, :self.P] - self.v)
        self.lam2 += self.gamma_ratio * (self.u[:, self.P:] - self.w)
        self.nu1 += self.gamma_ratio * (Gu1 - self.s)
        self.nu2 += self.gamma_ratio * (Gu2 - self.t)
        # Print elapsed time
        if verbose:
            print("Time for this iteration:", perf_counter() - ts)

    def ADMM_train(self, verbose=True, RBCD_verbose=True):
        t1 = perf_counter()
        self.set_dmat_emat()

        # Main training loop
        print('Starting ADMM iterations...')
        for k in range(self.iters):  # k is current iteration count
            _, _ = self.calc_loss_acc(k, verbose=verbose)
            self.ADMM_train_step(verbose=verbose, RBCD_verbose=RBCD_verbose)
            if verbose:
                print("Total time:", perf_counter() - t1)

        uu, alpha = self.calc_loss_acc(k + 1)
        return self.costs, self.costs2, self.dists, self.accuracies, self.v, self.w, uu, alpha
