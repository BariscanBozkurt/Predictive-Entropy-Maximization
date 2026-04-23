import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from bss.BSSbase import BSSBaseClass

class LDMIBSS(BSSBaseClass):

    def __init__(self,
                 n_sources: int,
                 presumed_domain: str = "antisparse",
                 method: str = "covariance", # can be either covariance or correlation
                 mu_y_start: float = 100.0,
                 mu_y_rule: str = "divide_by_root_index",
                 epsilon = 1e-5,
                 ### Ground truth source vectors. This part is only for debugging.
                 Sgt = None,
                 debug_iteration_point = 1000,
                 plot_debug_during_training = False,
                 ):
        self.n_sources = n_sources
        self.presumed_domain = presumed_domain
        if presumed_domain == "antisparse":
            self.project_onto_polytope = self.ProjectOntoLInfty
        elif presumed_domain == "nnantisparse":
            self.project_onto_polytope = self.ProjectOntoNNLInfty
        elif presumed_domain == "sparse":
            self.project_onto_polytope = self.ProjectRowstoL1NormBall
        elif presumed_domain == "nnsparse":
            self.project_onto_polytope = self.ProjectRowstoNNL1NormBall
        elif presumed_domain == "simplex":
            self.project_onto_polytope = self.ProjectRowstoUnitSimplex
        else:
            raise ValueError(f"Presumed domain '{presumed_domain}' not recognized.")
        self.method = method
        if method == "covariance":
            self.update_Y = self.update_Y_cov_based
            self.compute_corr = self.covariance
        elif method == "correlation":
            self.update_Y = self.update_Y_corr_based
            self.compute_corr = self.correlation
        
        self.mu_y_start = mu_y_start
        self.mu_y_rule = mu_y_rule
        self.epsilon = epsilon
        self.W = None

        self.Sgt = Sgt
        self.ground_truth_available = True if self.Sgt is not None else False
        self.component_SNR_history = [] # To track the SNR of the extracted components if ground truth is available
        self.SINR_history = [] # To track the SINR of the extracted components if ground truth is available
        self.SV_list = [] # To track the singular values of the feedforward weight for debugging and analysis of the learning dynamics
        self.debug_iteration_point = debug_iteration_point
        self.plot_debug_during_training = plot_debug_during_training

    @staticmethod
    @njit
    def correlation(X, Y = None):
        if Y is None:
            Y = X
        _, n_samples = X.shape
        RXY = (1 / n_samples) * np.dot(X, Y.T)
        return RXY
    
    @staticmethod
    @njit
    def covariance(X, Y = None):
        def mean_numba(a):

            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)
        if Y is None:
            Y = X
        _, n_samples = X.shape
        muY = mean_numba(Y).reshape(-1, 1)
        muX = mean_numba(X).reshape(-1, 1)
        RXY = (1 / n_samples) * (np.dot(X, Y.T) - np.outer(muX, muY))
        return RXY

    @staticmethod
    @njit
    def update_Y_corr_based(Y, X, W, epsilon, step_size):
        n_sources, samples = Y.shape[0], Y.shape[1]
        Identity_like_Y = np.eye(n_sources)
        RY = (1 / samples) * np.dot(Y, Y.T) + epsilon * Identity_like_Y
        E = Y - np.dot(W, X)
        RE = (1 / samples) * np.dot(E, E.T) + epsilon * Identity_like_Y
        gradY = (1 / samples) * (
            np.linalg.solve(RY, Y) - np.linalg.solve(RE, E)
        )
        Y = Y + (step_size) * gradY
        return Y

    @staticmethod
    @njit
    def update_Y_cov_based(Y, X, W, epsilon, step_size):
        def mean_numba(a):

            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)

        n_sources, samples = Y.shape[0], Y.shape[1]
        muY = mean_numba(Y).reshape(-1, 1)
        muX = mean_numba(X).reshape(-1, 1)
        Identity_like_Y = np.eye(n_sources)
        RY = (1 / samples) * (
            np.dot(Y, Y.T) - np.dot(muY, muY.T)
        ) + epsilon * Identity_like_Y
        E = (Y - muY) - np.dot(W, (X - muX.reshape(-1, 1)))
        muE = mean_numba(E).reshape(-1, 1)
        RE = (1 / samples) * (
            np.dot(E, E.T) - np.dot(muE, muE.T)
        ) + epsilon * Identity_like_Y
        gradY = (1 / samples) * (
            np.linalg.solve(RY, Y - muY) - np.linalg.solve(RE, E - muE)
        )
        Y = Y + (step_size) * gradY
        return Y

    def get_learning_rate(self, k: int):
        """
        Computes the learning rate based on the iteration index and specified rule.
        """
        if self.mu_y_rule == "divide_by_root_index":
            return self.mu_y_start / np.sqrt(k + 1)
        
        elif self.mu_y_rule == "divide_by_index":
            return self.mu_y_start / (k + 1)
        
        elif self.mu_y_rule == "exponential_decay":
            # Assuming a decay rate; you can add this to __init__
            decay_rate = 0.995
            return self.mu_y_start * (decay_rate ** k)
        
        elif self.mu_y_rule == "constant":
            return self.mu_y_start
            
        else:
            # Default to the most stable root decay
            return self.mu_y_start / np.sqrt(k + 1)

    def fit(self, X, n_iterations = 5000, regularize_W = False):
        X = np.ascontiguousarray(X)
        _, n_samples = X.shape
        if self.plot_debug_during_training:
            plt.figure(figsize=(45, 30), dpi=80)

        Y = np.random.rand(self.n_sources, n_samples) / 2
        Y = self.project_onto_polytope(Y.T).T
        # Y = np.zeros((self.n_sources, n_samples))
        RX = self.compute_corr(X)
        RXY = self.compute_corr(X, Y)
        if regularize_W:
            RX += self.epsilon * np.eye(RX.shape[0])
        W = np.linalg.solve(RX, RXY).T
        self.W = W
        for k in tqdm(range(n_iterations)):
            if self.ground_truth_available and k % self.debug_iteration_point == 0:
                Y_ = self.W @ X
                Y_ = self.signed_and_permutation_corrected_sources(self.Sgt, Y_) # Find sign and permutation ambiguity
                coef_ = ((Y_ * self.Sgt).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
                Y_ = coef_ * Y_
                self.component_SNR_history.append(self.ComputeSNR(self.Sgt, Y_))
                self.SINR_history.append(self.ComputeSINR(self.Sgt, Y_))
                self.SV_list.append(np.linalg.svd(self.W, compute_uv=False))
                if self.plot_debug_during_training:
                    self.plot_for_debug(self.SINR_history, self.component_SNR_history, self.debug_iteration_point, Y_[:, - 25 : -1].T)
            mu_y = self.get_learning_rate(k)
            Y = self.update_Y(Y, X, W, self.epsilon, mu_y)
            Y = self.project_onto_polytope(Y.T).T
            RXY = self.compute_corr(X, Y)
            W = np.linalg.solve(RX, RXY).T
            self.W = W
        
    def predict(self, X):
        return self.W @ X
