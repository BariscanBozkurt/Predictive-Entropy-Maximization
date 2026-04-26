import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python_utils.python_utils import Timer
from bss.bss_utils import generate_correlated_copula_sources, generate_uncorrelated_uniform_sources, addWGN, ProjectColstoSimplex, ProjectRowstoL1NormBall
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS

plt.rcParams.update({
    "font.size": 18,        # base font size
    "axes.titlesize": 22,   # title
    "axes.labelsize": 20,   # x/y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22
})

def get_mean_diag_history(Cy_history):
    """
    Convert C_y_history with shape (steps, n, n)
    into a 1D array of length steps:
    the mean of the diagonal entries at each step.
    """
    Cy_history = np.asarray(Cy_history)
    if Cy_history.ndim != 3:
        raise ValueError(f"Expected C_y_history shape (steps, n, n), got {Cy_history.shape}")

    n_steps, n, _ = Cy_history.shape
    diag_mask = np.eye(n, dtype=bool)
    mean_vals = np.mean(Cy_history[:, diag_mask], axis=1)
    return mean_vals


def run_single_experiment(seed, N=100000, NumberofSources=5, SNR=30):
    np.random.seed(seed)

    NumberofMixtures = NumberofSources + 5
    S = generate_correlated_copula_sources( rho=0, df=4,
                                                n_sources=NumberofSources, 
                                                size_sources=N, 
                                                decreasing_correlation=False)
    S = 2 * S - 1
    A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
    X_noNoise = np.dot(A, S)
    X = addWGN(X_noNoise, SNR)

    hyperparam_dict = {
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
                "debug_iteration_point" : 25000,
                "plot_debug_during_training" : False,
    }

    model = PredictiveDecorrBSS(**hyperparam_dict)
    model.fit(X)

    Y_ = model.predict(X)
    Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    snr_vals = np.array(model.ComputeSNR(S, Y_))  # shape: (n_sources,)

    if len(model.C_y_history) == 0:
        raise ValueError("C_y_history is empty. No debug points were recorded.")

    return get_mean_diag_history(model.C_y_history)


def plot_mean_and_sem(histories, title="Mean C_y evolution across experiments"):
    """
    histories: list of 1D arrays, one per experiment
    """
    histories = [np.asarray(h) for h in histories]

    min_len = min(len(h) for h in histories)
    histories = np.array([h[:min_len] for h in histories])  # align lengths

    mean_curve = np.mean(histories, axis=0)
    sem_curve = 2.045*np.std(histories, axis=0, ddof=1) / np.sqrt(histories.shape[0])

    x = np.arange(min_len)

    plt.figure(figsize=(12, 8))
    plt.plot(x, mean_curve, lw=2, label="Mean across experiments")
    plt.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve, alpha=0.25, label="± 95% CI")

    plt.xlabel("Sample")
    plt.ylabel("Mean Diagonal Covariance")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------
# Run multiple trials
# -------------------
n_experiments = 30
base_seed = 0
print("base seed is", base_seed)

all_histories = []
for i in range(n_experiments):
    seed = base_seed + i*100
    print(f"Running experiment {i+1}/{n_experiments} with seed {seed}")
    history = run_single_experiment(seed)
    all_histories.append(history)


histories = [np.asarray(h) for h in all_histories]
min_len = min(len(h) for h in histories)
histories = np.array([h[:min_len] for h in histories])

mean_curve = np.mean(histories, axis=0)

if min_len <= 20000:
    raise ValueError(f"History too short ({min_len}) to compute from timestep 20000 onward.")

final_mean_value = np.mean(mean_curve[20000:])
print("Mean of mean curve from timestep 20000 onward:", final_mean_value)

# -------------------

plot_mean_and_sem(all_histories, title="Evolution of the variance terms across experiments (Antisparse domain)")