import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from bss.BSSbase import BSSBaseClass

class BSMBSS(BSSBaseClass):

    def __init__(self, 
                n_sources,
                presumed_domain = "antisparse",
                ### Optimization parameters
                gamma = 0.999,
                eta = 1e-3,
                beta = 1e-7,
                ### Learning rates 
                neural_lr_start = 0.9,
                neural_lr_stop = 0.01,
                stlambda_lr = 0.5, # This is used in sparse and simplex domains for the soft-thresholding parameter update, it is not used in the antisparse domain
                neural_dynamics_iterations = 100,
                neural_OUTPUT_COMP_TOL = 1e-6,
                 ### Learning rate rules and decay parameters
                # lr_W_rule = "constant",
                # lr_W_decay_divider=5000,
                neural_lr_rule = "divide_by_loop_index",
                neural_lr_decay_divider=200,
                ### Initial values for weights if provided, if not they will be initialized in the fit function
                W = None,
                M = None,
                D = None,
                whiten_input_ =True,
                ### Ground truth source vectors. This part is only for debugging.
                Sgt = None,
                debug_iteration_point = 1000,
                plot_debug_during_training = False,
                ):
        if W is not None:
            assert W.shape[0] == n_sources, "The number of rows for the initial guess W must be n_sources=(%d)" % n_sources
        if M is not None:
            assert M.shape == (
                n_sources,
                n_sources,
            ), "The shape of the initial guess M must be (n_sources,n_sources)=(%d,%d)" % (n_sources, n_sources)
        if D is not None:
            assert D.shape == (
                n_sources,
                1,
            ), "The shape of the initial guess D must be (n_sources,1)=(%d,%d)" % (n_sources, 1)
        self.n_sources = n_sources
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        # if W is None:
        #     W = np.random.randn(n_sources, n_sources)
        #     W = 0.0033 * (
        #         W / np.sqrt(np.sum(np.abs(W) ** 2, axis=1)).reshape(n_sources, 1)
        #     )
        if D is None:
            D = np.ones((n_sources, 1))
        if M is None:
            M = np.eye(n_sources)

        self.Wpre = 1
        self.W = W
        self.M = M
        self.D = D
        self.neural_lr_start = neural_lr_start
        self.neural_lr_stop = neural_lr_stop
        self.stlambda_lr = stlambda_lr
        self.neural_dynamics_iterations = neural_dynamics_iterations
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.neural_lr_rule = neural_lr_rule
        self.neural_lr_decay_divider = neural_lr_decay_divider
        if presumed_domain == "antisparse":
            self.run_neural_dynamics = self.run_neural_dynamics_antisparse
        elif presumed_domain == "sparse":
            self.run_neural_dynamics = self.run_neural_dynamics_sparse
        else:
            raise ValueError(f"Presumed domain '{presumed_domain}' not recognized.")
        self.whiten_input_ = whiten_input_
        self.Sgt = Sgt
        self.ground_truth_available = True if self.Sgt is not None else False
        self.component_SNR_history = [] # To track the SNR of the extracted components if ground truth is available
        self.SINR_history = [] # To track the SINR of the extracted components if ground truth is available
        self.SV_list = [] # To track the singular values of the feedforward weight for debugging and analysis of the learning dynamics
        self.debug_iteration_point = debug_iteration_point
        self.plot_debug_during_training = plot_debug_during_training

    @staticmethod
    @njit
    def update_weights_jit(x_current, y, W, M, D, gamma, beta, eta):
        # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
        W = (gamma**2) * W + (1 - gamma**2) * np.outer(y, x_current)
        M = (gamma**2) * M + (1 - gamma**2) * np.outer(y, y)
        D = (1 - beta) * D + eta * (
            np.sum(np.abs(W) ** 2, axis=1) - np.sum((np.abs(M) ** 2) * D.T, axis=1)
        ).reshape(-1, 1)
        return W, M, D

    def compute_overall_mapping(self):
        W, M, D = self.W, self.M, self.D

        Wf = np.linalg.solve(M * D.T, W) @ self.Wpre
        self.Wf = Wf
        return Wf

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        x,
        y,
        W,
        M,
        D,
        neural_dynamics_iterations,
        neural_lr_start,
        neural_lr_stop,
        stlambd_lr = 0,
        lr_rule="divide_by_loop_index",
        lr_decay_divider=200,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def offdiag(A, return_diag=False):
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        M_hat, Upsilon = offdiag(M, True)

        u = Upsilon * ((D.T * y)[0])
        mat_factor1 = M_hat * D.T
        mat_factor2 = Upsilon * D.T
        # if fast_start:
        u = 0.99 * np.linalg.solve(M * D.T, W @ x)
        # u = W @ x
        y = np.clip(u / mat_factor2[0], -1, 1)
        for j in range(neural_dynamics_iterations):
            # Time-decaying step size (simulating cooling or annealing)
            if lr_rule == "constant":
                lr_y = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                lr_y = max(neural_lr_start / (j + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                lr_y = max(neural_lr_start / (j * lr_decay_divider + 1), neural_lr_stop)
            yold = y
            du = -u + (W @ x - mat_factor1 @ y)
            u = u - lr_y * du
            y = np.clip(u / mat_factor2[0], -1, 1)

            if np.linalg.norm(y - yold) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    #### Learning algorith is implemented with the fit function
    def fit(self, X, n_epochs=1):
        n_mixtures, n_samples = X.shape
        if self.plot_debug_during_training:
            plt.figure(figsize=(45, 30), dpi=80)
        idx = np.arange(n_samples)
        if self.W is None:
            if self.whiten_input_:
                self.W = np.eye(self.n_sources, self.n_sources)
            else:
                # Initialize W with small random values
                self.W = np.eye(self.n_sources, n_mixtures) + np.random.randn(self.n_sources, n_mixtures) * 0.01
                # W = np.random.randn(self.n_sources, n_mixtures)
                # W = 0.0033 * (
                #     W / np.sqrt(np.sum(np.abs(W) ** 2, axis=1)).reshape(self.n_sources, 1)
                # )
                # self.W = W
        if self.whiten_input_:
            X_white, Wpre = self.whiten_input(  X, 
                                                n_components = self.n_sources,
                                                return_prewhitening_matrix = True)
            self.Wpre = Wpre
        else:
            X_white = X
        for _ in range(n_epochs):
            for i_sample in tqdm(range(n_samples)):
                if self.ground_truth_available and i_sample % self.debug_iteration_point == 0:
                    _ = self.compute_overall_mapping()
                    Y_ = self.Wf @ X
                    Y_ = self.signed_and_permutation_corrected_sources(self.Sgt, Y_) # Find sign and permutation ambiguity
                    coef_ = ((Y_ * self.Sgt).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
                    Y_ = coef_ * Y_
                    self.component_SNR_history.append(self.ComputeSNR(self.Sgt, Y_))
                    self.SINR_history.append(self.ComputeSINR(self.Sgt, Y_))
                    self.SV_list.append(np.linalg.svd(self.Wf, compute_uv=False))
                    if self.plot_debug_during_training:
                        self.plot_for_debug(self.SINR_history, self.component_SNR_history, self.debug_iteration_point, Y_[:, idx[i_sample - 25 : i_sample]].T)
                # Randomly select a sample for online learning
                x_current = np.ascontiguousarray(X_white[:, idx[i_sample]])
                y = np.zeros(self.n_sources)
                # Run neural dynamics to find the optimal neural state for the current input
                y = self.run_neural_dynamics(x_current, y,
                                            self.W, self.M, self.D,
                                            self.neural_dynamics_iterations,
                                            self.neural_lr_start,
                                            self.neural_lr_stop,
                                            stlambd_lr = self.stlambda_lr,
                                            lr_rule = self.neural_lr_rule,
                                            lr_decay_divider = self.neural_lr_decay_divider,
                                            neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL)
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                self.W, self.M, self.D = self.update_weights_jit(
                    x_current, y, self.W, self.M, self.D, self.gamma, self.beta, self.eta
                )

    def predict(self, X):
        _ = self.compute_overall_mapping()
        return self.Wf @ X

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

