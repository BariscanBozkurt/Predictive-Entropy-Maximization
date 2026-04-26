import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
from python_utils.python_utils import Timer
from bss.bss_utils import generate_correlated_copula_sources, generate_uncorrelated_uniform_sources, addWGN, ProjectColstoSimplex, ProjectRowstoL1NormBall
from bss.PredictiveDecorrBSSSimple import PredictiveDecorrBSSSimple

def run_single(seed, N=100000, NumberofSources=5, SNR=30, gamma_lateral=1):
    np.random.seed(seed)

    NumberofMixtures = NumberofSources + 5
    S = generate_uncorrelated_uniform_sources(NumberofSources, N, min_val = -4, max_val = 4)
    S = ProjectColstoSimplex(S)

    A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
    X_noNoise = np.dot(A, S)
    X = addWGN(X_noNoise, SNR)

    hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "simplex",
                "gamma_lateral" : gamma_lateral,
                "epsilon" : 1e-4,
                ### Optimization parameters
                "lambda_lateral" : 0.99,
                "gamma_predictive" : 150,
                ### Learning rates 
                "lr_W" : 5 * 1e-2,
                "neural_lr_start" : 0.1,
                "neural_lr_stop" : 1e-4,
                "stlambda_lr" : 0.05,
                "neural_dynamics_iterations" : 100,
                "neural_OUTPUT_COMP_TOL" : 1e-7,
                ### Learning rate rules and decay parameters
                "lr_W_rule" : "divide_by_log_index",
                "lr_W_decay_divider" : 5000,
                "neural_lr_rule" : "divide_by_loop_index",
                "neural_lr_decay_divider" : 200,
                ### Initial values for weights if provided, if not they will be initialized in the fit function 
                "W" : None,
                "C_y" : None,
                "mu_y" : None, 
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 25000,
                "plot_debug_during_training" : False,
    }

    model = PredictiveDecorrBSSSimple(**hyperparam_dict)
    model.fit(X)

    Y_ = model.predict(X)
    Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    snr_vals = np.array(model.ComputeSNR(S, Y_))  # shape: (n_sources,)
    return snr_vals


# -----------------------
# Run experiments
# -----------------------
num_runs = 10
NumberofSources = 5

# Define the range of nn_antisparse_lateral_scaling values to search
scaling_values = [100,400,500,750]

base_seed = np.random.randint(5_000_000)
print("Base seed:", base_seed)

results = {}

for scaling in scaling_values:
    print(f"\nTesting gamma_lateral = {scaling}")
    all_snrs = np.zeros((num_runs, NumberofSources))
    
    for i in range(num_runs):
        all_snrs[i] = run_single(base_seed + i, NumberofSources=NumberofSources, gamma_lateral=scaling)
    
    mean_snr_per_channel = np.mean(all_snrs, axis=0)
    std_snr_per_channel = np.std(all_snrs, axis=0, ddof=1)
    
    results[scaling] = {
        'mean_snr': mean_snr_per_channel,
        'std_snr': std_snr_per_channel,
        'overall_mean': np.mean(mean_snr_per_channel)
    }

# -----------------------
# Final statistics
# -----------------------
print("\n=== Hyperparameter Search Results ===")
for scaling in scaling_values:
    res = results[scaling]
    print(f"\ngamma_lateral = {scaling}:")
    print(f"  Overall mean SNR: {res['overall_mean']:.4f}")
    for ch in range(NumberofSources):
        print(
            f"  Channel {ch}: mean SNR = {res['mean_snr'][ch]:.4f}, "
            f"std = {res['std_snr'][ch]:.4f}"
        )

# Find the best scaling
best_scaling = max(results, key=lambda x: results[x]['overall_mean'])
print(f"\nBest nn_antisparse_lateral_scaling: {best_scaling} with overall mean SNR: {results[best_scaling]['overall_mean']:.4f}")