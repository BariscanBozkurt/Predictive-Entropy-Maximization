import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from bss.BSSbase import BSSBaseClass

class PredictiveDecorrBSS(BSSBaseClass):

    def __init__(self,
                 n_sources, 
                 presumed_domain = "nnantisparse",
                 epsilon = 1e-5,
                 ### Optimization parameters
                 lambda_lateral = 0.99,
                 gamma_predictive = 100,
                 ### Learning rates 
                 lr_W = 0.01,
                 neural_lr_start = 0.9,
                 neural_lr_stop = 0.01,
                 stlambda_lr = 0.5, # This is used in sparse and simplex domains for the soft-thresholding parameter update, it is not used in the antisparse domain
                 neural_dynamics_iterations = 100,
                 neural_OUTPUT_COMP_TOL = 1e-8,
                 ### Learning rate rules and decay parameters
                 lr_W_rule = "constant",
                 lr_W_decay_divider=5000,
                 neural_lr_rule = "divide_by_loop_index",
                 neural_lr_decay_divider=200,
                 ### Initial values for weights if provided, if not they will be initialized in the fit function based on the input data dimension and n_components
                 W = None,
                 C_y = None,
                 mu_y = None, # Running mean of the extracted sources, used for the Taylor expansion of the log-determinant objective in the lateral inhibition term. This is important for non-stationary data where the source statistics may change over time.
                 ### Ground truth source vectors. This part is only for debugging.
                 Sgt = None,
                 debug_iteration_point = 1000,
                 plot_debug_during_training = False,
                 save_C_y_per_debug = False
                 ):
        self.n_sources = n_sources
        self.epsilon = epsilon  
        self.lambda_lateral = lambda_lateral
        self.gamma_predictive = gamma_predictive
        self.lr_W = lr_W
        self.neural_lr_start = neural_lr_start
        self.neural_lr_stop = neural_lr_stop
        self.stlambda_lr = stlambda_lr
        self.neural_dynamics_iterations = neural_dynamics_iterations
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.lr_W_rule = lr_W_rule
        self.lr_W_decay_divider = lr_W_decay_divider
        self.neural_lr_rule = neural_lr_rule
        self.neural_lr_decay_divider = neural_lr_decay_divider
        if presumed_domain == "antisparse":
            self.run_neural_dynamics = self.run_neural_dynamics_antisparse
        elif presumed_domain == "nnantisparse":
            self.run_neural_dynamics = self.run_neural_dynamics_nnantisparse
        elif presumed_domain == "sparse":
            self.run_neural_dynamics = self.run_neural_dynamics_sparse
        elif presumed_domain == "nnsparse":
            self.run_neural_dynamics = self.run_neural_dynamics_nnsparse
        elif presumed_domain == "simplex":
            self.run_neural_dynamics = self.run_neural_dynamics_simplex
        else:
            raise ValueError(f"Presumed domain '{presumed_domain}' not recognized.")
        self.W = W # Will be initialized in fit function based on the input data dimension and n_components
        if C_y is None:
            self.C_y = 0.2 * np.eye(n_sources, n_sources)
        else:
            self.C_y = C_y
        if mu_y is None:
            self.mu_y = np.zeros(n_sources)
        else:
            self.mu_y = mu_y

        self.Sgt = Sgt
        self.ground_truth_available = True if self.Sgt is not None else False
        self.component_SNR_history = [] # To track the SNR of the extracted components if ground truth is available
        self.SINR_history = [] # To track the SINR of the extracted components if ground truth is available
        self.SV_list = [] # To track the singular values of the feedforward weight for debugging and analysis of the learning dynamics
        self.debug_iteration_point = debug_iteration_point
        self.plot_debug_during_training = plot_debug_during_training
        self.save_C_y_per_debug = save_C_y_per_debug
        self.C_y_list = []
    #### Debugging functions for simulations if the ground truth source and mixing matrices are provide

    #### Neural dynamics algorithms for different source domains, e.g., sparse, simplex, etc.
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse( x, y,
                                        W, C_y, mu_y,
                                        gamma_predictive,
                                        epsilon,
                                        neural_dynamics_iterations,
                                        neural_lr_start,
                                        neural_lr_stop,
                                        stlambd_lr = 0,
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
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y) 
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon
        
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
            lateral = (np.dot(O_y, y_bar / (D_y + 1e-6)) - y_bar) / (D_y + 1e-6)
            
            # Combine gradients: Total force acting on the neural state
            grady = gamma_predictive * error + lateral
            
            # Gradient descent step
            y = y - lr_y * grady
            
            # Biological constraint: Firing rates are often bounded (activation function)
            y = np.clip(y, -1, 1)
            
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y
    
    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y,
                                        W, C_y, mu_y,
                                        gamma_predictive,
                                        epsilon,
                                        neural_dynamics_iterations,
                                        neural_lr_start,
                                        neural_lr_stop,
                                        stlambd_lr = 0,
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
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y) 
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon
        
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
            lateral = (np.dot(O_y, y_bar / (D_y + 1e-6)) - y_bar) / (D_y + 1e-6)
            
            # Combine gradients: Total force acting on the neural state
            grady = gamma_predictive * error + lateral
            
            # Gradient descent step
            y = y - lr_y * grady
            
            # Biological constraint: Firing rates are often bounded (activation function)
            y = np.clip(y, 0, 1)
            
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_sparse( x, y,
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
        STLAMBD = 0
        dval = 0
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y) 
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon
        
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
            y = y - lr_y * grady
            
            # Biological constraint: Firing rates are often bounded (activation function)
            # SOFT THRESHOLDING
            y_absolute = np.abs(y)
            y_sign = np.sign(y)

            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign
            dval = np.linalg.norm(y, 1) - 1
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)
            
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(   x, y,
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
        STLAMBD = 0
        dval = 0
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon
        
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
            y = y - lr_y * grady
            
            # Biological constraint: Firing rates are often bounded (activation function)
            # SOFT THRESHOLDING
            y = np.maximum(y - STLAMBD, 0)
            y = np.clip(y, 0, 5) # Ensure boundedness

            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)
            
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y
    
    @staticmethod
    @njit
    def run_neural_dynamics_simplex(x, y,
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
        STLAMBD = 0
        dval = 0
        # Pre-calculate the feedforward drive (projection of input into latent space)
        yke = np.dot(W, x)
        
        # Extract diagonal (variances) and off-diagonal (covariances) from L
        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
        D_y = D_y + epsilon
        
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
            y = y - lr_y * grady
            
            # Biological constraint: Firing rates are often bounded (activation function)
            # SOFT THRESHOLDING
            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = STLAMBD + stlambd_lr * dval
            
            # Convergence check: Exit if the activity has stabilized
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
                
        return y
    
    #### Learning algorith is implemented with the fit function
    def fit(self, X, n_epochs=1, shuffle_samples = False):
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
                if self.ground_truth_available and i_sample % self.debug_iteration_point == 0:
                    Y_ = self.W @ X
                    Y_ = self.signed_and_permutation_corrected_sources(self.Sgt, Y_) # Find sign and permutation ambiguity
                    coef_ = ((Y_ * self.Sgt).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
                    Y_ = coef_ * Y_
                    self.component_SNR_history.append(self.ComputeSNR(self.Sgt, Y_))
                    self.SINR_history.append(self.ComputeSINR(self.Sgt, Y_))
                    self.SV_list.append(np.linalg.svd(self.W, compute_uv=False))
                    if self.plot_debug_during_training:
                        self.plot_for_debug(self.SINR_history, self.component_SNR_history, self.debug_iteration_point, Y_[:, idx[i_sample - 25 : i_sample]].T)
                    if self.save_C_y_per_debug:
                        self.C_y_list.append(self.C_y)
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

    def predict(self, X):
        return self.W @ X

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
