import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
from numba import njit

from bss.bss_utils import generate_correlated_copula_sources, addWGN
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from python_utils.python_utils import Timer


class PredictiveDecorrBSSFullGrad(PredictiveDecorrBSS):
    """
    Same as PredictiveDecorrBSS, except that the antisparse neural dynamics
    use the full gradient of the surrogate objective, including the quadratic
    off-diagonal term discarded in the main biologically plausible version.
    """

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        x, y,
        W, C_y, mu_y,
        gamma_predictive,
        neural_dynamics_iterations,
        neural_lr_start,
        neural_lr_stop,
        stlambd_lr=0.0,
        lr_rule="divide_by_loop_index",
        lr_decay_divider=200,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        # Feedforward drive
        yke = np.dot(W, x)

        # Diagonal / off-diagonal decomposition
        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)

        eps = 1e-6
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
            lateral = (np.dot(O_y, y_bar / (D_y + eps)) - y_bar) / (D_y + eps)

            # Full quadratic correction term:
            # correction_k = ybar_k * sum_j O_kj^2 / (D_k^2 D_j)
            correction = np.zeros_like(y)
            for k in range(n):
                dk = D_y[k] + eps
                s = 0.0
                for jj in range(n):
                    dj = D_y[jj] + eps
                    o = O_y[k, jj]
                    s += (o * o) / (dk * dk * dj)
                correction[k] = y_bar[k] * s

            # Exact gradient = predictive + truncated lateral - correction
            grady = gamma_predictive * error + lateral - correction

            y = y - lr_y * grady
            y = np.clip(y, -1.0, 1.0)

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


print("Running script PredictiveBSS_Ablation_Truncated_vs_FullGradient_Antisparse")

results_dir = "../Results"
os.makedirs(results_dir, exist_ok=True)

csv_name_for_results = "predictive_bss_ablation_full_vs_truncated_gradient_antisparse_10by5_results.csv"
csv_path = os.path.join(results_dir, csv_name_for_results)

# --------------------------------------------------
# Simulation setup
# --------------------------------------------------
N = 100000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

target_SNRinp = 30
rho_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
seed_list = np.arange(0, 3000, 100)

predictivebss_hyperparam_dict = {
    "n_sources": NumberofSources,
    "presumed_domain": "antisparse",
    # Optimization parameters
    "lambda_lateral": 0.99,
    "gamma_predictive": 250,
    # Learning rates
    "lr_W": 5e-2,
    "neural_lr_start": 0.1,
    "neural_lr_stop": 1e-6,
    "neural_dynamics_iterations": 250,
    "neural_OUTPUT_COMP_TOL": 1e-7,
    # Learning-rate rules and decay parameters
    "lr_W_rule": "divide_by_index",
    "lr_W_decay_divider": 5000,
    "neural_lr_rule": "divide_by_loop_index",
    "neural_lr_decay_divider": 200,
    # # Initial values
    # "W": None,
    # "C_y": None,
    # "mu_y": None,
    # Debugging
    "debug_iteration_point": 25000,
    "plot_debug_during_training": False,
}

results_data = []

for rho in rho_list:
    for seed in seed_list:
        print(f"seed = {seed}, rho = {rho}")

        # --------------------------------------------------
        # Data generation
        # --------------------------------------------------
        np.random.seed(seed)

        S = generate_correlated_copula_sources(
            rho=rho,
            df=4,
            n_sources=NumberofSources,
            size_sources=N,
            decreasing_correlation=False,
        )
        S = 2 * S - 1

        A = np.random.randn(NumberofMixtures, NumberofSources)
        X_noNoise = A @ S
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp_measured = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X) ** 2, axis=1))
        )

        # --------------------------------------------------
        # Shared initialization for fair comparison
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
            "rho": float(rho),
            "SINR": SINR_result,
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp_measured),
            "execution_time": float(t.interval),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "antisparse",
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
            "rho": float(rho),
            "SINR": SINR_result,
            "SNRinp_target": float(target_SNRinp),
            "SNRinp_measured": float(SNRinp_measured),
            "execution_time": float(t.interval),
            "n_sources": int(NumberofSources),
            "n_mixtures": int(NumberofMixtures),
            "n_samples": int(N),
            "presumed_domain": "antisparse",
        }
        for k, snr_k in enumerate(SNR_result, start=1):
            result_dict_current[f"SNR_{k}"] = float(snr_k)
        results_data.append(result_dict_current)

        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(csv_path, index=False)

print(f"Saved results to: {csv_path}")