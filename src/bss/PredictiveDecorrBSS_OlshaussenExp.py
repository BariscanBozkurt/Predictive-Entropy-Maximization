import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Latex, Math, clear_output, display

import sys
sys.path.append("..")
from bss.BSSbase import BSSBaseClass
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS


class PredictiveDecorrOlshaussen(PredictiveDecorrBSS):
    def __init__(self,
                 n_sources, 
                 presumed_domain="sparse", # Usually sparse or nnsparse for patches
                 epsilon = 1e-5,
                 lambda_lateral=0.98,      # Slightly lower for non-stationary patches
                 gamma_predictive=10.0,    # Balanced based on our previous scaling talk
                 lr_W=0.02,
                 neural_lr_start=0.1,
                 neural_lr_stop=0.001,
                 stlambda_lr=0.1,          # Critical for enforcing the sparse penalty
                 neural_dynamics_iterations=100,
                 neural_OUTPUT_COMP_TOL=1e-7,
                 lr_W_rule="constant",
                 lr_W_decay_divider=5000,
                 neural_lr_rule="divide_by_loop_index",
                 neural_lr_decay_divider=200,
                 W=None,
                 C_y=None,
                 mu_y=None,
                 Sgt=None,
                 debug_iteration_point=1000,
                 plot_debug_during_training=False):
        
        # Initialize the parent class with all shared BSS parameters
        super().__init__(n_sources=n_sources,
                         presumed_domain=presumed_domain,
                         epsilon=epsilon,
                         lambda_lateral=lambda_lateral,
                         gamma_predictive=gamma_predictive,
                         lr_W=lr_W,
                         neural_lr_start=neural_lr_start,
                         neural_lr_stop=neural_lr_stop,
                         stlambda_lr=stlambda_lr,
                         neural_dynamics_iterations=neural_dynamics_iterations,
                         neural_OUTPUT_COMP_TOL=neural_OUTPUT_COMP_TOL,
                         lr_W_rule=lr_W_rule,
                         lr_W_decay_divider=lr_W_decay_divider,
                         neural_lr_rule=neural_lr_rule,
                         neural_lr_decay_divider=neural_lr_decay_divider,
                         W=W,
                         C_y=C_y,
                         mu_y=mu_y,
                         Sgt=Sgt,
                         debug_iteration_point=debug_iteration_point,
                         plot_debug_during_training=plot_debug_during_training)

        # Domain-specific initialization if not provided
        if C_y is None:
            # Olshausen filters often benefit from a smaller initial lateral penalty
            # to allow the feedforward weights to 'catch' features early on.
            self.C_y = 0.1 * np.eye(n_sources)

    @staticmethod
    @njit
    def run_neural_dynamics(x, y,
                            W, C_y, mu_y,
                            gamma_predictive,
                            epsilon,
                            neural_dynamics_iterations,
                            neural_lr_start,
                            neural_lr_stop,
                            stlambd_lr = 0.5,
                            lr_rule="divide_by_loop_index",
                            lr_decay_divider=200,
                            neural_OUTPUT_COMP_TOL=1e-7,
                            ):
        """
        Perform activity relaxation (inference) to find the optimal neural state 'y'.
        
        This function implements a fast-timescale gradient descent on the network 
        energy function. It balances two primary forces:
        1. Predictive Coding: Minimizing the error between the current state and 
        the feedforward input (y - Wx).
        2. Lateral Inhibition: Minimizing correlation 
        between neurons using a second-order Taylor approximation of the 
        log-determinant objective.

        Args:
            x (ndarray): Input signal vector.
            y (ndarray): Current neural activity vector (initial guess).
            W (ndarray): Feedforward weight matrix.
            C_y (ndarray): Lateral weight matrix (representing covariance/correlation).
            gamma_predictive (float): Weighting factor for the predictive error term.
            neural_dynamics_iterations (int): Maximum steps for activity relaxation.
            neural_lr_start (float): Initial step size for activity updates.
            neural_lr_stop (float): Minimum floor for the decaying step size.
            neural_OUTPUT_COMP_TOL (float): Convergence threshold for early exit.

        Returns:
            ndarray: The relaxed neural activity vector 'y'.
        """

        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded
        STLAMBD = 0
        dval = 0
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon  # Add epsilon for numerical stability in division
        
        for j in range(neural_dynamics_iterations):
            # Time-decaying step size (simulating cooling or annealing)
            if lr_rule == "constant":
                lr_y = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                lr_y = max(neural_lr_start / (j + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                lr_y = max(neural_lr_start / (j * lr_decay_divider + 1), neural_lr_stop)
            y_old = y.copy()
            
            # 1. Predictive Error Term: Measures mismatch between activity and input
            error = y - yke

            y_bar = y - mu_y
            # 2. Lateral Term: Implements normalized decorrelation
            lateral = (np.dot(O_y, y_bar / D_y) - y_bar) / D_y
            
            # Combine gradients: Total force acting on the neural state
            grady = gamma_predictive * error + lateral
            
            # Gradient descent step
            a = y - lr_y * grady
            # SOFT THRESHOLDING
            y = SoftThresholding(a, 0)
            # Biological constraint: Firing rates are often bounded (activation function)
            temp = 1
            if np.linalg.norm(a, 1) >= 1:
                iter2 = 0
                while ((np.abs(STLAMBD - temp) / np.abs(STLAMBD + 1e-10)) > 1e-5) & (iter2 < 10):
                    iter2 += 1
                    temp = STLAMBD

                    y = SoftThresholding(a, STLAMBD)
                    sstep = stlambd_lr / np.sqrt(iter2)
                    dval = np.linalg.norm(y,1) - 1
                    STLAMBD = STLAMBD + sstep * dval
                    if STLAMBD < 0:
                        STLAMBD = 0
                        y = a
                # y = np.clip(y, -1, 1)
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y

    def plot_receptive_fields(self, W, patch_size=(12, 12), title="Learned Filters"):
        """
        Plots the feedforward weights as a grid of 2D receptive fields.
        
        Args:
            W (ndarray): Weight matrix of shape (n_filters, n_pixels)
            patch_size (tuple): The (height, width) of the original image patches.
        """
        n_filters = W.shape[0]
        
        # 1. Calculate grid dimensions (closest to a square)
        n_cols = int(np.ceil(np.sqrt(n_filters)))
        n_rows = int(np.ceil(n_filters / n_cols))
        
        # 2. Setup figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        fig.suptitle(title, fontsize=20)
        
        # Flatten axes array for easy iteration if it's 2D
        axes_flat = axes.flatten()
        
        for i in range(n_filters):
            # Reshape the i-th row into a 2D patch
            rf = W[i, :].reshape(patch_size)
                        
            ax = axes_flat[i]
            ax.imshow(rf, cmap='gray', interpolation='nearest')
            ax.axis('off')
            
        # 3. Hide any unused subplot axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.9)
        
        clear_output(wait=True)
        display(plt.gcf())
        plt.close() # Prevents memory leaks by closing the figure after display
        
    def fit(self, X, n_epochs=1, shuffle_samples = True):
        n_mixtures, n_samples = X.shape
        if self.plot_debug_during_training:
            plt.figure(figsize=(45, 30), dpi=80)
        if shuffle_samples:
            idx = np.random.permutation(n_samples)
        else:
            idx = np.arange(n_samples)
        if self.W is None:
            # Initialize W with small random values
            self.W = np.eye(self.n_sources, n_mixtures) + np.random.randn(self.n_sources, n_mixtures) * 0.01
        for epoch in range(n_epochs):
            for i_sample in tqdm(range(n_samples)):
                if i_sample % self.debug_iteration_point == 0:
                    self.plot_receptive_fields(self.W, title=f"Learned Filters at Epoch {epoch}, Sample {i_sample}")
                # Randomly select a sample for online learning
                x_current = np.ascontiguousarray(X[:, idx[i_sample]])
                y = np.zeros(self.n_sources)
                # Run neural dynamics to find the optimal neural state for the current input
                y = self.run_neural_dynamics(x_current, y,
                                            self.W, self.C_y, self.mu_y,
                                            self.gamma_predictive,
                                            self.epsilon,
                                            self.neural_dynamics_iterations,
                                            self.neural_lr_start,
                                            self.neural_lr_stop,
                                            stlambd_lr = self.stlambda_lr,
                                            lr_rule = self.neural_lr_rule,
                                            lr_decay_divider = self.neural_lr_decay_divider,
                                            neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL)
                # Update the feedforward weights based on the current neural state and input
                # Here we use a simple Hebbian update rule modulated by the predictive error
                error = y - self.W @ x_current
                if self.lr_W_rule == "constant":
                    lr_W = self.lr_W
                elif self.lr_W_rule == "divide_by_log_index":
                    lr_W = max(self.lr_W / (1 + np.log((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 2)), 1e-8) # Decay learning rate over time
                elif self.lr_W_rule == "divide_by_index":
                    lr_W = max(self.lr_W / ((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 1), 1e-8)
                self.W += lr_W * np.outer(error, x_current) # Hebbian update with predictive error modulation
                # Update the running mean of the extracted sources for the Taylor expansion in the lateral inhibition term
                self.mu_y = self.lambda_lateral * self.mu_y + (1 - self.lambda_lateral) * y # Exponential moving average to track the mean of the extracted sources
                y_bar = y - self.mu_y
                self.C_y = self.lambda_lateral * self.C_y + (1 - self.lambda_lateral) * np.outer(y_bar, y_bar)
