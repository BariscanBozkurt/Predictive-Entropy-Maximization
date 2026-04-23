import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import generate_correlated_copula_sources, addWGN
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


def sample_mixing_matrix(dist_name, n_rows, n_cols, rng, max_tries=1000):
    """
    Sample a full-column-rank mixing matrix with iid entries from a chosen distribution.
    All distributions are scaled to have approximately zero mean and unit variance.
    """
    for _ in range(max_tries):
        if dist_name == "gaussian":
            A = rng.standard_normal((n_rows, n_cols))

        elif dist_name == "uniform":
            # Var(U[-a,a]) = a^2 / 3, choose a = sqrt(3) for unit variance
            A = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n_rows, n_cols))

        elif dist_name == "laplace":
            # Var(Laplace(0,b)) = 2 b^2, choose b = 1/sqrt(2) for unit variance
            A = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(n_rows, n_cols))

        elif dist_name == "rademacher":
            A = rng.choice(np.array([-1.0, 1.0]), size=(n_rows, n_cols))

        elif dist_name == "student_t_df5":
            # Var(t_df) = df/(df-2), so multiply by sqrt((df-2)/df) for unit variance
            df = 5.0
            A = rng.standard_t(df, size=(n_rows, n_cols)) * np.sqrt((df - 2.0) / df)

        else:
            raise ValueError(f"Unknown mixing distribution: {dist_name}")

        if np.linalg.matrix_rank(A) == n_cols:
            return A

    raise RuntimeError(
        f"Could not sample a full-column-rank mixing matrix for distribution '{dist_name}' "
        f"after {max_tries} attempts."
    )


print("Running script PredictiveBSS_Ablation_MixingDistribution_NNAntisparse_10by5")

results_dir = "../Results"
os.makedirs(results_dir, exist_ok=True)

csv_name_for_results = "predictive_bss_ablation_mixing_distribution_nnantisparse_10by5_results.csv"
csv_path = os.path.join(results_dir, csv_name_for_results)

# --------------------------------------------------
# Simulation setup
# --------------------------------------------------
N = 100000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

rho = 0.0
target_SNRinp = 30

seed_list = np.arange(0, 3000, 100)
mixing_distribution_list = [
    "gaussian",
    "uniform",
    "laplace",
    "rademacher",
    "student_t_df5",
]

predictivebss_hyperparam_dict = {
    "n_sources": NumberofSources,
    "presumed_domain": "nnantisparse",
    # Optimization parameters
    "lambda_lateral": 0.95,
    "gamma_predictive": 750,
    # Learning rates
    "lr_W": 5e-2,
    "neural_lr_start": 0.05,
    "neural_lr_stop": 1e-4,
    "neural_dynamics_iterations": 500,
    "neural_OUTPUT_COMP_TOL": 1e-6,
    # Learning-rate rules and decay parameters
    "lr_W_rule": "divide_by_index",
    "lr_W_decay_divider": 20000,
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

for dist_idx, mixing_distribution in enumerate(mixing_distribution_list):
    for seed in seed_list:
        print(f"seed = {seed}, mixing_distribution = {mixing_distribution}")

        # --------------------------------------------------
        # Generate sources and noisy mixtures
        # Keep source/noise realization fixed across distributions for a given seed.
        # Only the mixing matrix distribution changes.
        # --------------------------------------------------
        np.random.seed(seed)

        S = generate_correlated_copula_sources(
            rho=rho,
            df=4,
            n_sources=NumberofSources,
            size_sources=N,
            decreasing_correlation=False,
        )

        rng_mix = np.random.default_rng(seed + 100000 * (dist_idx + 1))
        A = sample_mixing_matrix(
            mixing_distribution,
            NumberofMixtures,
            NumberofSources,
            rng_mix,
        )

        X_noNoise = A @ S
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X) ** 2, axis=1))
        )

        # --------------------------------------------------
        # Run PredictiveDecorr
        # Keep initialization fixed across distributions for a given seed.
        # --------------------------------------------------
        print("Running PredictiveDecorrBSS")
        with Timer() as t:
            model = PredictiveDecorrBSS(
                **predictivebss_hyperparam_dict,
                Sgt=S,
            )

            model.C_y = 2.0 * np.eye(NumberofSources)

            rng_W = np.random.default_rng(seed + 200000)
            model.W = (
                rng_W.standard_normal((NumberofSources, NumberofMixtures)) / 15.0
                + np.eye(NumberofSources, NumberofMixtures) * 0.01
            )

            model.fit(X)

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR_result = float(model.ComputeSINR(S, Y_))
        SNR_result = np.asarray(model.ComputeSNR(S, Y_), dtype=float).ravel()

        print(f"SINR: {SINR_result}")
        print(f"Component SNRs: {SNR_result}\n")

        result_dict_current = {
            "Model": "PredictiveDecorrBSS",
            "seed": int(seed),
            "rho": float(rho),
            "mixing_distribution": mixing_distribution,
            "SINR": SINR_result,
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp),
            "execution_time": float(t.interval),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "nnantisparse",
        }

        for k, snr_k in enumerate(SNR_result, start=1):
            result_dict_current[f"SNR_{k}"] = float(snr_k)

        results_data.append(result_dict_current)

        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(csv_path, index=False)

print(f"Saved results to: {csv_path}")