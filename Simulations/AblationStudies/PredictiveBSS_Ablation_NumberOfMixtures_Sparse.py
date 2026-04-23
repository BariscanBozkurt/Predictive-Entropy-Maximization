import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import (
    generate_uncorrelated_uniform_sources,
    addWGN,
    ProjectRowstoL1NormBall,
)
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


print("Running script PredictiveBSS_Ablation_NumberOfMixtures_Sparse")

results_dir = "../Results"
os.makedirs(results_dir, exist_ok=True)

csv_name_for_results = "predictive_bss_ablation_number_of_mixtures_sparse_results.csv"
csv_path = os.path.join(results_dir, csv_name_for_results)

# --------------------------------------------------
# Simulation setup
# --------------------------------------------------
N = 100000
NumberofSources = 5
NumberofMixtures_list = [7, 8, 9, 10, 11, 12, 13]

target_SNRinp = 30.0
seed_list = np.arange(0, 3000, 100)

predictivebss_hyperparam_dict = {
    "n_sources": NumberofSources,
    "presumed_domain": "sparse",
    # Optimization parameters
    "lambda_lateral": 0.99,
    "gamma_predictive": 150,
    # Learning rates
    "lr_W": 5e-2,
    "neural_lr_start": 0.05,
    "neural_lr_stop": 1e-4,
    "stlambda_lr": 0.5,
    "neural_dynamics_iterations": 100,
    "neural_OUTPUT_COMP_TOL": 1e-6,
    # Learning-rate rules and decay parameters
    "lr_W_rule": "divide_by_index",
    "lr_W_decay_divider": 5000,
    "neural_lr_rule": "divide_by_loop_index",
    "neural_lr_decay_divider": 200,
    # Initial values
    "W": None,
    "C_y": None,
    "mu_y": None,
    # Debugging
    "debug_iteration_point": 25000,
    "plot_debug_during_training": False,
}

results_data = []

for seed in seed_list:
    print(f"seed = {seed}")

    # --------------------------------------------------
    # Generate one source realization per seed
    # --------------------------------------------------
    np.random.seed(seed)
    S = generate_uncorrelated_uniform_sources(
        NumberofSources, N, min_val=-4, max_val=4
    )
    S = ProjectRowstoL1NormBall(S.T).T

    # --------------------------------------------------
    # Generate one "max-size" Gaussian mixing matrix per seed.
    # Then use the first m rows for each mixture count.
    # --------------------------------------------------
    np.random.seed(seed + 1)
    A_max = np.random.randn(max(NumberofMixtures_list), NumberofSources)

    for NumberofMixtures in NumberofMixtures_list:
        print(f"  n_mixtures = {NumberofMixtures}")

        A = A_max[:NumberofMixtures, :]
        X_noNoise = A @ S

        # --------------------------------------------------
        # Add noise at fixed input SNR
        # --------------------------------------------------
        np.random.seed(seed + 1000 + NumberofMixtures)
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp_measured = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X) ** 2, axis=1))
        )

        # --------------------------------------------------
        # Run PredictiveDecorr
        # --------------------------------------------------
        print("    Running PredictiveDecorrBSS")
        np.random.seed(seed + 2000 + NumberofMixtures)

        with Timer() as t:
            model = PredictiveDecorrBSS(
                **predictivebss_hyperparam_dict,
                Sgt=S,
            )
            model.fit(X)

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR_result = float(model.ComputeSINR(S, Y_))
        SNR_result = np.asarray(model.ComputeSNR(S, Y_), dtype=float).ravel()

        print(f"    SINR: {SINR_result}")
        print(f"    Component SNRs: {SNR_result}\n")

        result_dict_current = {
            "Model": "PredictiveDecorrBSS",
            "seed": int(seed),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "sparse",
            "SINR": SINR_result,
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp_measured),
            "execution_time": float(t.interval),
        }

        for k, snr_k in enumerate(SNR_result, start=1):
            result_dict_current[f"SNR_{k}"] = float(snr_k)

        results_data.append(result_dict_current)

        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(csv_path, index=False)

print(f"Saved results to: {csv_path}")