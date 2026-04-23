import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import (
    generate_uncorrelated_uniform_sources,
    addWGN,
    ProjectColstoSimplex,
)
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


def sample_nested_gaussian_mixing_matrix(n_rows_max, n_cols, prefix_row_counts, rng, max_tries=1000):
    """
    Sample one Gaussian mixing matrix A_max of size (n_rows_max, n_cols) such that
    every prefix A_max[:m, :] for m in prefix_row_counts has full column rank.
    """
    for _ in range(max_tries):
        A_max = rng.standard_normal((n_rows_max, n_cols))
        ok = True
        for m in prefix_row_counts:
            if np.linalg.matrix_rank(A_max[:m, :]) < n_cols:
                ok = False
                break
        if ok:
            return A_max

    raise RuntimeError(
        "Could not sample a nested full-column-rank Gaussian mixing matrix "
        f"after {max_tries} attempts."
    )


print("Running script PredictiveBSS_Ablation_NumberOfMixtures_Simplex")

results_dir = "../Results"
os.makedirs(results_dir, exist_ok=True)

csv_name_for_results = "predictive_bss_ablation_number_of_mixtures_simplex_results.csv"
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
    "presumed_domain": "simplex",
    # Optimization parameters
    "lambda_lateral": 0.99,
    "gamma_predictive": 150,
    # Learning rates
    "lr_W": 5e-2,
    "neural_lr_start": 0.1,
    "neural_lr_stop": 1e-4,
    "stlambda_lr": 0.05,
    "neural_dynamics_iterations": 100,
    "neural_OUTPUT_COMP_TOL": 1e-7,
    # Learning-rate rules and decay parameters
    "lr_W_rule": "divide_by_log_index",
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
    # Generate one simplex source realization per seed
    # --------------------------------------------------
    np.random.seed(seed)
    S = generate_uncorrelated_uniform_sources(
        NumberofSources, N, min_val=-4, max_val=4
    )
    S = ProjectColstoSimplex(S)

    # --------------------------------------------------
    # Generate one nested Gaussian mixing matrix per seed
    # so that all tested mixture counts use prefixes of the same A
    # --------------------------------------------------
    rng_mix = np.random.default_rng(seed + 1)
    A_max = sample_nested_gaussian_mixing_matrix(
        n_rows_max=max(NumberofMixtures_list),
        n_cols=NumberofSources,
        prefix_row_counts=NumberofMixtures_list,
        rng=rng_mix,
    )

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
            "presumed_domain": "simplex",
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