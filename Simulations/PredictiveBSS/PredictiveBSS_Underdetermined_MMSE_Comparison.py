import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from bss.bss_utils import generate_uncorrelated_uniform_sources, addWGN
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


def align_sources_one_to_one(S, Y):
    """Match PEM outputs bijectively, then fit one signed LS gain per source."""
    S_centered = S - S.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    denominator = np.outer(np.linalg.norm(S_centered, axis=1),
                           np.linalg.norm(Y_centered, axis=1))
    correlation = np.divide(S_centered @ Y_centered.T, denominator,
                            out=np.zeros_like(denominator), where=denominator > 0)
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    permutation = np.empty(S.shape[0], dtype=int)
    permutation[rows] = columns
    matched = Y[permutation]
    energy = np.sum(matched ** 2, axis=1)
    gains = np.divide(np.sum(S * matched, axis=1), energy,
                      out=np.zeros_like(energy), where=energy > 0)
    return gains[:, None] * matched


def component_snr(S, Y):
    """Source-indexed waveform SNR after the explicitly chosen alignment."""
    return 10 * np.log10(np.sum(S ** 2, axis=1) /
                         np.sum((S - Y) ** 2, axis=1))


print("Running script PredictiveBSS_Underdetermined_MMSE_Comparison")
if not os.path.exists("../Results"):
    os.mkdir("../Results")

# Preserve the historical CSV; verified evaluation has a separate destination.
csv_name_for_results = "predictive_bss_underdetermined_mmse_comparison_verified_results.csv"

### Setting of the simulation.
N = 100000
NumberofSources = 5
target_SNRinp = 30

### Predictive Entropy Maximization (PEM) model hyperparameters (antisparse source domain).
predictivebss_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "antisparse",
                ### Optimization parameters
                "lambda_lateral" : 0.999,
                "gamma_predictive" : 250,
                ### Learning rates
                "lr_W" : 5 * 1e-2,
                "neural_lr_start" : 0.5,
                "neural_lr_stop" : 1e-6,
                "neural_dynamics_iterations" : 250,
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

### We consider the UNDERDETERMINED case: fewer mixtures than sources.
number_of_mixtures_list = [4, 3]
### We run the experiment over 30 different random mixing matrices (30 seeds).
seed_list = np.arange(0, 30)

results_data = []
for NumberofMixtures in number_of_mixtures_list:
    for seed in seed_list:
        np.random.seed(seed)
        print("Number of mixtures is {}, seed is {}".format(NumberofMixtures, seed))
        ##################################################
        ##### GENERATE SOURCES AND MIX THEM ##############
        ##################################################
        # Antisparse sources, uniformly distributed in [-1, 1].
        S = generate_uncorrelated_uniform_sources(NumberofSources, N, min_val = -1, max_val = 1)
        # Random Gaussian mixing matrix. This is underdetermined (NumberofMixtures < NumberofSources).
        A = np.random.randn(NumberofMixtures, NumberofSources)
        X_noNoise = np.dot(A, S)

        # Add white Gaussian noise so that the input signal-to-noise ratio is the target value.
        X = addWGN(X_noNoise, target_SNRinp)
        NoisePart = X - X_noNoise

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X)**2, axis=1))
        )

        ##################################################
        ####### PREDICTIVE ENTROPY MAXIMIZATION ##########
        ##################################################
        print("Running Predictive BSS Model")
        with Timer() as t:
            model = PredictiveDecorrBSS(**predictivebss_hyperparam_dict,
                                        Sgt = S)
            model.fit(X, n_epochs=1, shuffle_samples=False)

        Y_pem = align_sources_one_to_one(S, model.predict(X))
        pem_component_snr = component_snr(S, Y_pem)
        print("PEM Component Signal-to-Noise-Ratio (SNR) Values : {}".format(pem_component_snr))

        ##################################################
        ####### EMPIRICAL LINEAR-MMSE REFERENCE #########
        ##################################################
        # Ground-truth-assisted, zero-intercept least squares on the SAME evaluation samples.
        # These are empirical second moments, not known population covariances. The reference
        # minimizes each source's in-sample linear-prediction error. Its rows already correspond
        # to the physical sources, so do not rematch, flip, or rescale them.
        R_xx = (X @ X.T) / N
        R_sx = (S @ X.T) / N
        W_mmse = np.linalg.solve(R_xx, R_sx.T).T
        Y_mmse = W_mmse @ X
        mmse_component_snr = component_snr(S, Y_mmse)
        assert np.isfinite(pem_component_snr).all() and np.isfinite(mmse_component_snr).all()
        assert np.all(pem_component_snr <= mmse_component_snr + 1e-8)
        print("Empirical LMMSE Component SNR Values : {}\n".format(mmse_component_snr))

        ##################################################
        ####### SAVE THE PER-SOURCE RESULTS ##############
        ##################################################
        # We store one row per source so that the visualization notebook can sort the per-source
        # SNR values independently within each method/trial and aggregate over 30 seeds.
        for source_index in range(NumberofSources):
            result_dict_current = {
                'num_sources': NumberofSources,
                'num_mixtures': NumberofMixtures,
                'seed': seed,
                'source_index': source_index,
                'pem_component_snr': pem_component_snr[source_index],
                'ls_reference_component_snr': mmse_component_snr[source_index],
                'input_snr': SNRinp,
                'pem_execution_time': t.interval,
            }
            results_data.append(result_dict_current)

        # Save incrementally so that partial results are not lost if the run is interrupted.
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(os.path.join("../Results", csv_name_for_results), index=False)

print("Done. Results saved to {}".format(os.path.join("../Results", csv_name_for_results)))
