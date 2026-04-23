import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
from numba import njit

from bss.bss_utils import (
    generate_uncorrelated_uniform_sources,
    addWGN,
    ProjectRowstoL1NormBall,
)
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


class PredictiveDecorrBSSFullGrad(PredictiveDecorrBSS):
    """
    Same as PredictiveDecorrBSS, except that the sparse neural dynamics
    use the full gradient of the surrogate objective, including the
    quadratic off-diagonal term discarded in the main biologically
    plausible version.
    """

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(
        x, y,
        W, C_y, mu_y,
        gamma_predictive,
        epsilon,
        neural_dynamics_iterations,
        neural_lr_start,
        neural_lr_stop,
        stlambd_lr=0.5,
        lr_rule="divide_by_loop_index",
        lr_decay_divider=200,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        STLAMBD = 0.0
        dval = 0.0

        # Feedforward drive
        yke = np.dot(W, x)

        # Diagonal / off-diagonal decomposition
        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
        D_y += epsilon
        n = y.shape[0]

        for j in range(neural_dynamics_iterations):
            if lr_rule == "constant":
                lr_y = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                lr_y = max(neural_lr_start / (j + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                lr_y = max(neural_lr_start / (j * lr_decay_divider + 1), neural_lr_stop)
            else:
                lr_y = neural_lr_start

            y_old = y.copy()

            # Predictive term
            error = y - yke

            # Truncated lateral term
            y_bar = y - mu_y
            lateral = (np.dot(O_y, y_bar / D_y) - y_bar) / D_y

            # Full quadratic correction term:
            # correction_k = ybar_k * sum_j O_kj^2 / (D_k^2 D_j)
            correction = np.zeros_like(y)
            for k in range(n):
                dk = D_y[k]
                s = 0.0
                for jj in range(n):
                    dj = D_y[jj]
                    o = O_y[k, jj]
                    s += (o * o) / (dk * dk * dj)
                correction[k] = y_bar[k] * s

            # Exact gradient = predictive + truncated lateral - correction
            grady = gamma_predictive * error + lateral - correction

            # Gradient descent step
            y = y - lr_y * grady

            # Same sparse proximal step as in the original implementation
            y_absolute = np.abs(y)
            y_sign = np.sign(y)
            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign

            dval = np.linalg.norm(y, 1) - 1.0
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0.0)

            # Convergence check
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break

        return y


def evaluate_model(model, S, X):
    Y_ = model.predict(X)
    Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    SINR_result = float(model.ComputeSINR(S, Y_))
    SNR_result = np.asarray(model.ComputeSNR(S, Y_), dtype=float).ravel()
    return SINR_result, SNR_result


print("Running script PredictiveBSS_Ablation_Truncated_vs_FullGradient_Sparse")

results_dir = "../Results"
os.makedirs(results_dir, exist_ok=True)

csv_name_for_results = "predictive_bss_ablation_full_vs_truncated_gradient_sparse_10by5_results.csv"
csv_path = os.path.join(results_dir, csv_name_for_results)

# --------------------------------------------------
# Simulation setup
# --------------------------------------------------
N = 100000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

input_snr_list = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0]
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
    # IMPORTANT:
    # Do NOT include "W", "C_y", "mu_y" here, since we pass them explicitly below
    # for a fair comparison between truncated and full gradients.
    "debug_iteration_point": 25000,
    "plot_debug_during_training": False,
}

results_data = []

for snr_ in input_snr_list:
    for seed in seed_list:
        print(f"seed = {seed}, input_snr = {snr_}")

        # --------------------------------------------------
        # Data generation
        # --------------------------------------------------
        np.random.seed(seed)

        S = generate_uncorrelated_uniform_sources(
            NumberofSources, N, min_val=-4, max_val=4
        )
        S = ProjectRowstoL1NormBall(S.T).T

        A = np.random.randn(NumberofMixtures, NumberofSources)
        X_noNoise = A @ S

        target_SNRinp = snr_
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp_measured = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X) ** 2, axis=1))
        )

        # --------------------------------------------------
        # Shared initialization for fair comparison
        # Match the default initialization in PredictiveDecorrBSS
        # --------------------------------------------------
        rng_init = np.random.default_rng(seed + 123456)
        W_init = (
            np.eye(NumberofSources, NumberofMixtures)
            + rng_init.standard_normal((NumberofSources, NumberofMixtures)) * 0.01
        )
        C_y_init = 0.2 * np.eye(NumberofSources, NumberofSources)
        mu_y_init = np.zeros(NumberofSources)

        # ==================================================
        # 1) Truncated/noisy gradient version
        # ==================================================
        print("  Running PredictiveDecorrBSS (truncated gradient)")
        with Timer() as t:
            model = PredictiveDecorrBSS(
                **predictivebss_hyperparam_dict,
                W=W_init.copy(),
                C_y=C_y_init.copy(),
                mu_y=mu_y_init.copy(),
                Sgt=S,
            )
            model.fit(X)

        SINR_result, SNR_result = evaluate_model(model, S, X)

        result_dict_current = {
            "Model": "PredictiveDecorrBSS",
            "gradient_variant": "truncated",
            "seed": int(seed),
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp_measured),
            "SINR": SINR_result,
            "execution_time": float(t.interval),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "sparse",
        }
        for k, snr_k in enumerate(SNR_result, start=1):
            result_dict_current[f"SNR_{k}"] = float(snr_k)
        results_data.append(result_dict_current)

        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(csv_path, index=False)

        # ==================================================
        # 2) Full-gradient version
        # ==================================================
        print("  Running PredictiveDecorrBSSFullGrad (full gradient)")
        with Timer() as t:
            model = PredictiveDecorrBSSFullGrad(
                **predictivebss_hyperparam_dict,
                W=W_init.copy(),
                C_y=C_y_init.copy(),
                mu_y=mu_y_init.copy(),
                Sgt=S,
            )
            model.fit(X)

        SINR_result, SNR_result = evaluate_model(model, S, X)

        result_dict_current = {
            "Model": "PredictiveDecorrBSS",
            "gradient_variant": "full",
            "seed": int(seed),
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp_measured),
            "SINR": SINR_result,
            "execution_time": float(t.interval),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "sparse",
        }
        for k, snr_k in enumerate(SNR_result, start=1):
            result_dict_current[f"SNR_{k}"] = float(snr_k)
        results_data.append(result_dict_current)

        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(csv_path, index=False)

print(f"Saved results to: {csv_path}")