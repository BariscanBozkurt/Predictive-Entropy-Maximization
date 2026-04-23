import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import generate_correlated_copula_sources, addWGN
from bss.BSSbase import BSSBaseClass
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from bss.CorInfoMaxBSS import OnlineCorInfomax
from bss.LDMIBSS import LDMIBSS
from bss.ica_utils import fit_icainfomax
from bss.BSMBSS import BSMBSS
from python_utils.python_utils import Timer

def evaluate_taylor_surrogate(C_y):
    """
    Evaluates the exact log-determinant, its 2nd-order Taylor surrogate, 
    and the theoretical error upper bound for a given covariance matrix.

    Args:
        C_y (np.ndarray): A symmetric, positive-definite covariance matrix of shape (n, n).

    Returns:
        tuple: (original_energy, approximate_energy, error_upper_bound)
    """
    # 1. Original Energy Term (Exact Log-Determinant)
    sign, logabsdet = np.linalg.slogdet(C_y)
    original_energy = sign * logabsdet

    # 2. Approximate Energy Term (2nd-Order Taylor Surrogate)
    D_y_vec = np.diag(C_y)
    O_y = C_y - np.diag(D_y_vec)
    
    # Calculate D^{-1} O using broadcasting (faster and more numerically stable than np.linalg.inv)
    trace_inside_term = (1.0 / D_y_vec.reshape(-1, 1)) * O_y
    
    approximate_energy = np.sum(np.log(D_y_vec)) - 0.5 * np.linalg.trace(trace_inside_term @ trace_inside_term)

    # 3. Theoretical Error Upper Bound
    O_frob_sq = np.linalg.norm(O_y, ord='fro')**2
    O_spectral = np.linalg.norm(O_y, ord=2)
    
    # Minimum eigenvalue of C_y (using eigvalsh since C_y is a symmetric matrix)
    lambda_min_C = np.min(np.linalg.eigvalsh(C_y))
    
    # Calculate the analytical bound
    error_upper_bound = (1.0 / 3.0) * (O_frob_sq * O_spectral) / (lambda_min_C**3)

    return original_energy, approximate_energy, error_upper_bound

def evaluate_taylor_surrogate_batch(model_C_y_list):
    """
    Iterates over a sequence of covariance matrices to compute the Taylor 
    surrogate metrics for each matrix.

    Args:
        model_C_y_list (list): A list of symmetric, positive-definite covariance 
                               matrices (np.ndarray) recorded during training.

    Returns:
        tuple: A tuple containing three lists:
            - actual_errors (list): The absolute difference between the exact 
              log-determinant and the 2nd-order Taylor approximation.
            - theoretical_bounds (list): The analytically derived upper bound 
              for the approximation error.
            - off_diagonal_norms (list): The Frobenius norm of the off-diagonal 
              matrix O_y for each time step.
            (Note: A small epsilon of 1e-16 is added to errors and bounds to 
            prevent log(0) issues during plotting).
    """
    actual_errors = []
    theoretical_bounds = []
    off_diagonal_norms = []
    
    for C_y in model_C_y_list:
        # 1. Get metrics from our previously defined function
        orig_energy, approx_energy, bound = evaluate_taylor_surrogate(C_y)
        actual_error = np.abs(orig_energy - approx_energy)
        
        # 2. Calculate the Frobenius norm of the off-diagonal matrix O_y
        D_y_vec = np.diag(C_y)
        O_y = C_y - np.diag(D_y_vec)
        O_frob_norm = np.linalg.norm(O_y, ord='fro')
        
        # 3. Store the results
        # We add a tiny epsilon (1e-16) to avoid log(0) warnings in the plot if they hit exactly 0
        actual_errors.append(actual_error + 1e-16)
        theoretical_bounds.append(bound + 1e-16)
        off_diagonal_norms.append(O_frob_norm)
        
    return actual_errors, theoretical_bounds, off_diagonal_norms

print("Running script PredictiveBSS_Correlated_Antisparse_10by5")
if not os.path.exists("../Results"):
    os.mkdir("../Results")

# RESULTS_DF = pd.DataFrame( columns = ['seed', 'Model', 'SINR', 'SNR', 'SNRinp', 'execution_time'])
csv_name_for_results = "predictive_bss_antisparse_taylor_error_results_V2.csv"
### Setting of the simulation and the model hyperparameters.
N = 40000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

predictivebss_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "antisparse",
                ### Optimization parameters
                "lambda_lateral" : 0.99,
                "gamma_predictive" : 250,
                ### Learning rates 
                "lr_W" : 5 * 1e-2,
                "neural_lr_start" : 0.5,
                "neural_lr_stop" : 1e-6,
                "neural_dynamics_iterations" : 250,
                "neural_OUTPUT_COMP_TOL" : 1e-7,
                ### Learning rate rules and decay parameters
                "lr_W_rule" : "divide_by_index",
                "lr_W_decay_divider" : 5000,
                "neural_lr_rule" : "divide_by_loop_index",
                "neural_lr_decay_divider" : 200,
                ### Initial values for weights if provided, if not they will be initialized in the fit function 
                "W" : None,
                "C_y" : None,
                "mu_y" : None, 
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 1000,
                "plot_debug_during_training" : False,
                "save_C_y_per_debug" : True,
}

rho_list = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
seed_list = np.arange(0, 3000, 100)
results_data = []
for rho in rho_list:
    for seed in seed_list:
        np.random.seed(seed)
        print("seed is ", seed)
        ##################################################
        ##### GENERATE SOURCES ###########################
        ##################################################
        # # Generate Sources and Mix Them (10 by 5 case)
        S = generate_correlated_copula_sources( rho=rho, df=4,
                                                n_sources=NumberofSources, 
                                                size_sources=N, 
                                                decreasing_correlation=False)
        S = 2 * S - 1
        A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
        X_noNoise = np.dot(A, S)

        target_SNRinp = 30
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X)**2, axis=1))
        )

        ##################################################
        ####### PREDICTIVE BSS ###########################
        ##################################################
        print("Running Predictive BSS Model")
        with Timer() as t:
            model = PredictiveDecorrBSS(**predictivebss_hyperparam_dict,
                                        Sgt = S)
            C_y_sq = np.random.randn(NumberofSources, NumberofSources) / 5 + np.sqrt(0.2)*np.eye(NumberofSources)
            model.C_y = C_y_sq @ C_y_sq.T
            model.fit(X)

        # In[7]:

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_) # Find sign and permutation ambiguity
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
        Y_ = coef_ * Y_

        SINR_result = model.ComputeSINR(S, Y_)
        SNR_result = model.ComputeSNR(S, Y_)
        print("Signal-to-Interference-and-Noise-Ratio (SINR): {}".format(SINR_result))
        print("Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(SNR_result))

        actual_errors, theoretical_bounds, off_diagonal_norms = evaluate_taylor_surrogate_batch(model.C_y_list)

        result_dict_current = {
            'Model': 'PredictiveDecorrBSS',
            'seed': seed,
            'rho' : rho,
            'SINR': SINR_result,
            'SNR': [SNR_result],
            'SNRinp': target_SNRinp,
            'actual_error' : [np.array(actual_errors)],
            'theoretical_bounds' : [np.array(theoretical_bounds)],
            'off_diagonal_norms' : [np.array(off_diagonal_norms)],
            'execution_time': t.interval
        }
        results_data.append(result_dict_current)
        result_df_current = pd.DataFrame(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(os.path.join("../Results", csv_name_for_results))

