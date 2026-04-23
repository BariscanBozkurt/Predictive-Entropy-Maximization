import os
import sys
import json

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


def serialize_array(array_like):
    """Serialize a 1D array-like object as a JSON string for CSV storage."""
    return json.dumps(np.asarray(array_like, dtype=float).tolist())


def evaluate_taylor_surrogate(C_y, eig_tol=1e-12):
    """
    Evaluate the exact log-determinant, its 2nd-order Taylor surrogate,
    the tighter spectral upper bound, and the original loose upper bound.

    Parameters
    ----------
    C_y : np.ndarray
        Symmetric positive definite covariance matrix of shape (n, n).
    eig_tol : float
        Numerical safety floor for denominators involving 1 + lambda_i.

    Returns
    -------
    dict
        Dictionary containing:
            - original_energy
            - approximate_energy
            - signed_remainder
            - abs_remainder
            - tight_bound
            - loose_bound
            - off_diagonal_frob_norm
            - B_frob_norm
            - B_spectral_norm
            - lambda_min_C
            - lambda_min_B
    """
    # Symmetrize for numerical safety
    C_y = 0.5 * (C_y + C_y.T)

    # Exact log-det
    sign, logabsdet = np.linalg.slogdet(C_y)
    if sign <= 0:
        raise ValueError("C_y must be positive definite to evaluate log det.")
    original_energy = logabsdet

    # Diagonal / off-diagonal decomposition
    D_y_vec = np.diag(C_y).copy()
    if np.any(D_y_vec <= 0):
        raise ValueError("Diagonal entries of C_y must be strictly positive.")
    O_y = C_y - np.diag(D_y_vec)

    # Normalized off-diagonal matrix B = D^{-1/2} O D^{-1/2}
    D_inv_sqrt = 1.0 / np.sqrt(D_y_vec)
    B = (D_inv_sqrt[:, None] * O_y) * D_inv_sqrt[None, :]
    B = 0.5 * (B + B.T)  # keep symmetric numerically

    eigvals_B = np.linalg.eigvalsh(B)
    lambda_min_B = np.min(eigvals_B)

    if lambda_min_B <= -1.0:
        raise ValueError(
            f"Encountered lambda_min(B) = {lambda_min_B:.6e} <= -1. "
            "This violates the SPD condition of I + B."
        )

    # 2nd-order Taylor surrogate
    # log det(C) ≈ sum_i log D_ii - 1/2 * Tr(B^2)
    approximate_energy = np.sum(np.log(D_y_vec)) - 0.5 * np.sum(eigvals_B**2)

    # Exact signed remainder
    signed_remainder = original_energy - approximate_energy
    abs_remainder = np.abs(signed_remainder)

    # -------- Tighter spectral bound --------
    # R_2 <= max( (1/3) sum_{lambda_i >= 0} lambda_i^3,
    #             (1/3) sum_{lambda_i < 0} |lambda_i|^3 / (1 + lambda_i) )
    pos_eigs = eigvals_B[eigvals_B >= 0.0]
    neg_eigs = eigvals_B[eigvals_B < 0.0]

    positive_part = (1.0 / 3.0) * np.sum(pos_eigs**3)

    if neg_eigs.size > 0:
        denom = np.maximum(1.0 + neg_eigs, eig_tol)
        negative_part = (1.0 / 3.0) * np.sum((np.abs(neg_eigs) ** 3) / denom)
    else:
        negative_part = 0.0

    tight_bound = max(positive_part, negative_part)

    # -------- Original loose bound --------
    O_frob_sq = np.linalg.norm(O_y, ord="fro") ** 2
    O_spectral = np.linalg.norm(O_y, ord=2)
    lambda_min_C = np.min(np.linalg.eigvalsh(C_y))

    if lambda_min_C <= 0:
        raise ValueError("C_y must be positive definite for the loose bound.")

    loose_bound = (1.0 / 3.0) * (O_frob_sq * O_spectral) / (lambda_min_C ** 3)

    # Diagnostics
    off_diagonal_frob_norm = np.linalg.norm(O_y, ord="fro")
    B_frob_norm = np.linalg.norm(B, ord="fro")
    B_spectral_norm = np.linalg.norm(B, ord=2)

    return {
        "original_energy": original_energy,
        "approximate_energy": approximate_energy,
        "signed_remainder": signed_remainder,
        "abs_remainder": abs_remainder,
        "tight_bound": tight_bound,
        "loose_bound": loose_bound,
        "off_diagonal_frob_norm": off_diagonal_frob_norm,
        "B_frob_norm": B_frob_norm,
        "B_spectral_norm": B_spectral_norm,
        "lambda_min_C": lambda_min_C,
        "lambda_min_B": lambda_min_B,
    }


def evaluate_taylor_surrogate_batch(model_C_y_list):
    """
    Evaluate Taylor surrogate diagnostics across a sequence of covariance matrices.

    Parameters
    ----------
    model_C_y_list : list[np.ndarray]
        List of covariance matrices recorded during training.

    Returns
    -------
    dict
        Dictionary of time-series lists:
            - abs_remainders
            - signed_remainders
            - tight_bounds
            - loose_bounds
            - off_diagonal_frob_norms
            - B_frob_norms
            - B_spectral_norms
            - lambda_min_C_list
            - lambda_min_B_list
    """
    abs_remainders = []
    signed_remainders = []
    tight_bounds = []
    loose_bounds = []
    off_diagonal_frob_norms = []
    B_frob_norms = []
    B_spectral_norms = []
    lambda_min_C_list = []
    lambda_min_B_list = []

    for C_y in model_C_y_list:
        metrics = evaluate_taylor_surrogate(C_y)

        abs_remainders.append(metrics["abs_remainder"])
        signed_remainders.append(metrics["signed_remainder"])
        tight_bounds.append(metrics["tight_bound"])
        loose_bounds.append(metrics["loose_bound"])
        off_diagonal_frob_norms.append(metrics["off_diagonal_frob_norm"])
        B_frob_norms.append(metrics["B_frob_norm"])
        B_spectral_norms.append(metrics["B_spectral_norm"])
        lambda_min_C_list.append(metrics["lambda_min_C"])
        lambda_min_B_list.append(metrics["lambda_min_B"])

    return {
        "abs_remainders": abs_remainders,
        "signed_remainders": signed_remainders,
        "tight_bounds": tight_bounds,
        "loose_bounds": loose_bounds,
        "off_diagonal_frob_norms": off_diagonal_frob_norms,
        "B_frob_norms": B_frob_norms,
        "B_spectral_norms": B_spectral_norms,
        "lambda_min_C_list": lambda_min_C_list,
        "lambda_min_B_list": lambda_min_B_list,
    }


print("Running script PredictiveBSS_Correlated_Antisparse_10by5")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

csv_name_for_results = "predictive_bss_antisparse_taylor_error_results_V3.csv"

### Setting of the simulation and the model hyperparameters.
N = 40000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

predictivebss_hyperparam_dict = {
    "n_sources": NumberofSources,
    "presumed_domain": "antisparse",
    ### Optimization parameters
    "lambda_lateral": 0.99,
    "gamma_predictive": 250,
    ### Learning rates
    "lr_W": 5 * 1e-2,
    "neural_lr_start": 0.5,
    "neural_lr_stop": 1e-6,
    "neural_dynamics_iterations": 250,
    "neural_OUTPUT_COMP_TOL": 1e-7,
    ### Learning rate rules and decay parameters
    "lr_W_rule": "divide_by_index",
    "lr_W_decay_divider": 5000,
    "neural_lr_rule": "divide_by_loop_index",
    "neural_lr_decay_divider": 200,
    ### Initial values for weights if provided, if not they will be initialized in the fit function
    "W": None,
    "C_y": None,
    "mu_y": None,
    ### Ground truth source vectors. This part is only for debugging.
    "debug_iteration_point": 1000,
    "plot_debug_during_training": False,
    "save_C_y_per_debug": True,
}

rho_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
seed_list = np.arange(0, 3000, 100)

results_data = []

for rho in rho_list:
    for seed in seed_list:
        np.random.seed(seed)
        print("seed is ", seed)

        ##################################################
        ##### GENERATE SOURCES ###########################
        ##################################################
        S = generate_correlated_copula_sources(
            rho=rho,
            df=4,
            n_sources=NumberofSources,
            size_sources=N,
            decreasing_correlation=False,
        )
        S = 2 * S - 1

        A = np.random.randn(NumberofMixtures, NumberofSources)
        X_noNoise = np.dot(A, S)

        target_SNRinp = 30
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X) ** 2, axis=1))
        )

        ##################################################
        ####### PREDICTIVE BSS ###########################
        ##################################################
        print("Running Predictive BSS Model")
        with Timer() as t:
            model = PredictiveDecorrBSS(
                **predictivebss_hyperparam_dict,
                Sgt=S,
            )
            C_y_sq = (
                np.random.randn(NumberofSources, NumberofSources) / 5
                + np.sqrt(0.2) * np.eye(NumberofSources)
            )
            model.C_y = C_y_sq @ C_y_sq.T
            model.fit(X)

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_)
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR_result = model.ComputeSINR(S, Y_)
        SNR_result = model.ComputeSNR(S, Y_)

        print(f"Signal-to-Interference-and-Noise-Ratio (SINR): {SINR_result}")
        print(f"Component Signal-to-Noise-Ratio (SNR) Values : {SNR_result}\n")

        taylor_metrics = evaluate_taylor_surrogate_batch(model.C_y_list)

        result_dict_current = {
            "Model": "PredictiveDecorrBSS",
            "seed": seed,
            "rho": rho,
            "SINR": SINR_result,
            "SNR": serialize_array(SNR_result),
            "target_SNRinp": target_SNRinp,
            "measured_SNRinp": SNRinp,
            "actual_error_abs": serialize_array(taylor_metrics["abs_remainders"]),
            "actual_remainder_signed": serialize_array(taylor_metrics["signed_remainders"]),
            "tight_theoretical_bound": serialize_array(taylor_metrics["tight_bounds"]),
            "loose_theoretical_bound": serialize_array(taylor_metrics["loose_bounds"]),
            "off_diagonal_frob_norm": serialize_array(taylor_metrics["off_diagonal_frob_norms"]),
            "B_frob_norm": serialize_array(taylor_metrics["B_frob_norms"]),
            "B_spectral_norm": serialize_array(taylor_metrics["B_spectral_norms"]),
            "lambda_min_C": serialize_array(taylor_metrics["lambda_min_C_list"]),
            "lambda_min_B": serialize_array(taylor_metrics["lambda_min_B_list"]),
            "execution_time": t.interval,
        }

        results_data.append(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_csv(os.path.join("../Results", csv_name_for_results), index=False)