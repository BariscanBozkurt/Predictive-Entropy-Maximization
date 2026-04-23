import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from bss.BSSbase import BSSBaseClass

# Assuming inheritance from BSSBaseClass as in BSMBSS
class OnlineNSM(BSSBaseClass):
    def __init__(self, 
                n_sources,
                whiten_input_=True,
                neural_dynamics_iterations=250,
                W1=None,
                W2=None,
                Dt=None,
                Sgt=None,
                debug_iteration_point=1000,
                plot_debug_during_training=False):
        
        self.n_sources = n_sources
        self.whiten_input_ = whiten_input_
        self.neural_dynamics_iterations = neural_dynamics_iterations
        
        # Initial Weights: If None, they will be initialized in the fit function
        self.W1 = W1
        self.W2 = np.zeros((n_sources, n_sources)) if W2 is None else W2
        self.Dt = 0.1 * np.ones((n_sources, 1)) if Dt is None else Dt
        
        self.Wpre = 1.0 # Default identity pre-whitening
        self.Sgt = Sgt
        self.ground_truth_available = True if Sgt is not None else False
        
        self.debug_iteration_point = debug_iteration_point
        self.plot_debug_during_training = plot_debug_during_training
        
        # Statistics tracking
        self.component_SNR_history = []
        self.SINR_history = []
        self.SV_list = []

    @staticmethod
    @njit
    def run_neural_dynamics(x, y, W1, W2, n_iterations):
        """Coordinate descent neural dynamics with ReLU activation."""
        n_sources = y.shape[0]
        # Ensure x and y are the right shape for dot products inside njit
        # We use flattening or indexing to ensure scalar/vector consistency
        for _ in range(n_iterations):
            # Numba-friendly random integer selection
            ind = np.random.randint(0, n_sources)
            
            # Compute the net input to the selected neuron
            # W1[ind, :] @ x (feedforward) - W2[ind, :] @ y (lateral)
            # We ensure y is viewed as a flat array for the dot product
            gate = np.dot(W1[ind, :], x.flatten()) - np.dot(W2[ind, :], y.flatten())
            
            # Apply ReLU and update in-place
            # np.maximum(scalar, scalar) works well in nopython mode
            y[ind, 0] = np.maximum(gate, 0.0)
            
        return y

    # @staticmethod
    # @njit
    # def update_weights_jit(xk, y, W1, W2, Dt, n_sources):
    #     """NSM Weight updates normalized by running variance Dt."""
    #     # Update running variance with a ceiling for numerical stability
    #     Dt = np.minimum(3000.0, 0.94 * Dt + y**2)
        
    #     invDt = 1.0 / Dt.flatten()
        
    #     # W1 (Feedforward) update
    #     term1_W1 = np.outer(y.flatten(), xk.flatten())
    #     term2_W1 = np.diag(y.flatten()**2) @ W1
    #     W1 += (invDt.reshape(-1, 1)) * (term1_W1 - term2_W1)
        
    #     # W2 (Lateral) update
    #     term1_W2 = np.outer(y.flatten(), y.flatten())
    #     term2_W2 = np.diag(y.flatten()**2) @ W2
    #     W2 += (invDt.reshape(-1, 1)) * (term1_W2 - term2_W2)
        
    #     # Ensure 0 diagonal for W2 (no self-inhibition in this version)
    #     for i in range(n_sources):
    #         W2[i, i] = 0.0
            
    #     return W1, W2, Dt

    def compute_overall_mapping(self):
        """Reconstructs the global filter from learned weights."""
        # Solve steady state (I + W2)y = W1*Wpre*x
        self.Wf = np.linalg.solve(np.eye(self.n_sources) + self.W2, self.W1) @ self.Wpre
        return self.Wf

    def fit(self, X, n_epochs=1):
        n_mixtures, n_samples = X.shape
        s_dim ,x_dim = self.n_sources, n_mixtures
        ZERO_CHECK_INTERVAL = 1500
        nzerocount = np.zeros(self.n_sources)
        if self.plot_debug_during_training:
            plt.figure(figsize=(45, 30), dpi=80)
        # Whitening logic (Inside fit, similar to BSMBSS)
        if self.whiten_input_:
            X_input, self.Wpre = self.whiten_input(X, n_components=self.n_sources, return_prewhitening_matrix=True)
            x_dim = self.n_sources  # After whitening, the input dimension matches n_sources
        else:
            X_input = X
            self.Wpre = np.eye(n_mixtures)

        # Initialize W1 if not provided (now that we know n_mixtures)
        if self.W1 is None:
            input_dim = X_input.shape[0]
            self.W1 = np.eye(self.n_sources, input_dim)

        for epoch in range(n_epochs):
            idx = np.random.permutation(n_samples)
            
            for i_sample in tqdm(range(n_samples)):
                Dt, W1, W2 = self.Dt, self.W1, self.W2
                xk = X_input[:, idx[i_sample]].reshape(-1, 1)
                
                # Evaluation/Debug logic
                if self.ground_truth_available and i_sample % self.debug_iteration_point == 0:
                    Wf = self.compute_overall_mapping()
                    Y_est = Wf @ X
                    Y_est = self.signed_and_permutation_corrected_sources(self.Sgt, Y_est)
                    coef_ = ((Y_est * self.Sgt).sum(axis=1) / (Y_est * Y_est).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
                    Y_est = coef_ * Y_est
                    self.component_SNR_history.append(self.ComputeSNR(self.Sgt, Y_est))
                    self.SINR_history.append(self.ComputeSINR(self.Sgt, Y_est))
                    self.SV_list.append(np.linalg.svd(Wf, compute_uv=False))
                    
                    if self.plot_debug_during_training:
                        self.plot_for_debug(self.SINR_history, self.component_SNR_history, 
                                          self.debug_iteration_point, 
                                          Y_est[:, idx[i_sample-25:i_sample]].T)

                # Inference
                y = np.random.rand(self.n_sources, 1)
                y = self.run_neural_dynamics(xk, y, self.W1, self.W2, self.neural_dynamics_iterations)
                
                # Update Weights
                Dt = np.minimum(3000, 0.94 * Dt + y**2)
                DtD = np.diag(1 / Dt.reshape(s_dim))
                W1 = W1 + np.dot(
                    DtD,
                    (
                        np.dot(y, (xk.T).reshape((1, x_dim)))
                        - np.dot(np.diag((y**2).reshape(s_dim)), W1)
                    ),
                )
                W2 = W2 + np.dot(
                    DtD,
                    (np.dot(y, y.T) - np.dot(np.diag((y**2).reshape(s_dim)), W2)),
                )

                for ind in range(s_dim):
                    W2[ind, ind] = 0

                nzerocount = (nzerocount + (y.reshape(s_dim) == 0) * 1.0) * (
                    y.reshape(s_dim) == 0
                )
                if i_sample < ZERO_CHECK_INTERVAL:
                    q = np.argwhere(nzerocount > 50)
                    qq = q[:, 0]
                    for iter3 in range(len(qq)):
                        W1[qq[iter3], :] = -W1[qq[iter3], :]
                        nzerocount[qq[iter3]] = 0

                self.W1 = W1
                self.W2 = W2
                self.Dt = Dt

    def predict(self, X):
        Wf = self.compute_overall_mapping()
        return Wf @ X