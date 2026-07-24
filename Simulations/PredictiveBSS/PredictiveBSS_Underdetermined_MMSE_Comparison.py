import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import generate_uncorrelated_uniform_sources, addWGN
from bss.BSSbase import BSSBaseClass
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer

print("Running script PredictiveBSS_Underdetermined_MMSE_Comparison")
if not os.path.exists("../Results"):
    os.mkdir("../Results")

csv_name_for_results = "predictive_bss_underdetermined_mmse_comparison_results.csv"

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
### We run the experiment over 10 different random mixing matrices (10 seeds).
seed_list = np.arange(0, 10)

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
            model.fit(X)

        Y_pem = model.predict(X)
        Y_pem = model.signed_and_permutation_corrected_sources(S, Y_pem) # Find sign and permutation ambiguity
        coef_pem = ((Y_pem * S).sum(axis=1) / (Y_pem * Y_pem).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
        Y_pem = coef_pem * Y_pem

        pem_overall_sinr = model.ComputeSINR(S, Y_pem)
        pem_component_snr = model.ComputeSNR(S, Y_pem)
        print("PEM Component Signal-to-Noise-Ratio (SNR) Values : {}".format(pem_component_snr))

        ##################################################
        ####### MMSE (WIENER) ORACLE ESTIMATOR ###########
        ##################################################
        # The MMSE (Wiener) estimator is the best achievable LINEAR reconstruction.
        # It knows the mixing matrix A, the source covariance R_s, and the noise covariance R_n.
        # In the underdetermined case this is the fundamental ceiling for any linear separator (PEM included).
        R_s = (S @ S.T) / N                       # source covariance matrix
        R_n = (NoisePart @ NoisePart.T) / N       # noise covariance matrix
        W_mmse = R_s @ A.T @ np.linalg.inv(A @ R_s @ A.T + R_n)   # Wiener filter (NumberofSources x NumberofMixtures)
        Y_mmse = W_mmse @ X

        Y_mmse = BSSBaseClass().signed_and_permutation_corrected_sources(S, Y_mmse) # Find sign and permutation ambiguity
        coef_mmse = ((Y_mmse * S).sum(axis=1) / (Y_mmse * Y_mmse).sum(axis=1)).reshape(-1, 1) # amplitude correction (same as PEM)
        Y_mmse = coef_mmse * Y_mmse

        mmse_overall_sinr = BSSBaseClass().ComputeSINR(S, Y_mmse)
        mmse_component_snr = BSSBaseClass().ComputeSNR(S, Y_mmse)
        print("MMSE Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(mmse_component_snr))

        ##################################################
        ####### SAVE THE PER-SOURCE RESULTS ##############
        ##################################################
        # We store one row per source so that the visualization notebook can sort the per-source
        # SNR values (e.g. by recoverability) and average them over the 10 random mixing matrices.
        for source_index in range(NumberofSources):
            result_dict_current = {
                'num_sources': NumberofSources,
                'num_mixtures': NumberofMixtures,
                'seed': seed,
                'source_index': source_index,
                'pem_component_snr': pem_component_snr[source_index],
                'mmse_component_snr': mmse_component_snr[source_index],
                'pem_overall_sinr': pem_overall_sinr,
                'mmse_overall_sinr': mmse_overall_sinr,
                'input_snr': SNRinp,
                'pem_execution_time': t.interval,
            }
            results_data.append(result_dict_current)

        # Save incrementally so that partial results are not lost if the run is interrupted.
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(os.path.join("../Results", csv_name_for_results), index=False)

print("Done. Results saved to {}".format(os.path.join("../Results", csv_name_for_results)))
