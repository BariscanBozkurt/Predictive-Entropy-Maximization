import os
import sys
sys.path.append("../../src")

import pandas as pd
import numpy as np
from numba import njit
from scipy.stats import ortho_group
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import pywt

from python_utils.visualization_utils import display_matrix, plot_bss_comparison, subplot_1D_signals
from python_utils.dsp_utils import transform_to_wavelet, reconstruct_from_wavelet
from bss.bss_utils import generate_uncorrelated_uniform_sources, addWGN
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS

def load_audio_source(name, duration=5, sr=16000):
    # Load example clips from librosa
    path = librosa.ex(name)
    y, _ = librosa.load(path, duration=duration, sr=sr)
    return y

print("Running script PredictiveBSS_SoundSeparation_Multiple")
if not os.path.exists("../Results"):
    os.mkdir("../Results")

# RESULTS_DF = pd.DataFrame( columns = ['seed', 'Model', 'SINR', 'SNR', 'SNRinp', 'execution_time'])
pickle_name_for_results = "predictive_bss_sound_separation_5by3_results.pkl"

seed_list = np.arange(0, 3000, 100)
results_data = []
for seed_ in seed_list:
    np.random.seed(seed_)

    # Load 3 distinct types of sound
    s1 = load_audio_source('fishin') 
    s2 = load_audio_source('pistachio')
    s3 = load_audio_source('vibeace')  

    # Stack into S (n_sources, T)
    # Ensure they are all the same length
    min_len = min(len(s1), len(s2), len(s3))
    S = np.stack([s1[:min_len], s2[:min_len], s3[:min_len]]).astype(np.float64)
    print("Length of the sound signals are ", min_len)
    # This only required for debugging
    wavelet_type = 'db4' # Daubechies 4 is excellent for audio/images
    S_wavelet, slices = transform_to_wavelet(S, wavelet=wavelet_type, level=3)

    S /= np.max(np.abs(S_wavelet), axis = 1).reshape(-1, 1)
    print("The following is the correlation matrix of sources")
    display_matrix(np.corrcoef(S))

    NumberofSources = S.shape[0]
    NumberofMixtures = 5
    # # Generate Mxr random mixing from i.i.d N(0,1)
    A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
    # A = ortho_group.rvs(dim=NumberofSources) # Random orthogonal mixing matrix
    X_noNoise = np.dot(A, S)


    target_SNRinp = 30
    X = addWGN(X_noNoise, target_SNRinp)

    SNRinp = 10 * np.log10(
        np.sum(np.mean(X_noNoise ** 2, axis=1))
        / np.sum(np.mean((X_noNoise - X)**2, axis=1))
    )
    print("The following is the mixture matrix A")
    print("Input SNR is : {}".format(SNRinp))

    # 1. Transform Mixtures to Sparse Domain
    wavelet_type = 'db4' # Daubechies 4 is excellent for audio/images
    X_wavelet, slices = transform_to_wavelet(X, wavelet=wavelet_type, level=3)

    hyperparam_dict = {
                    "n_sources" :  NumberofSources,
                    "presumed_domain" : "sparse",
                    ### Optimization parameters
                    "lambda_lateral" : 0.95,
                    "gamma_predictive" : 150,
                    ### Learning rates 
                    "lr_W" : 9.5 * 1e-1,
                    "neural_lr_start" : 0.01,
                    "neural_lr_stop" : 1e-4,
                    "stlambda_lr" : 0.5,
                    "neural_dynamics_iterations" : 100,
                    "neural_OUTPUT_COMP_TOL" : 1e-6,
                    ### Learning rate rules and decay parameters
                    "lr_W_rule" : "divide_by_index",
                    "lr_W_decay_divider" : 2000,
                    "neural_lr_rule" : "divide_by_loop_index",
                    "neural_lr_decay_divider" : 200,
                    ### Initial values for weights if provided, if not they will be initialized in the fit function 
                    "W" : None,
                    "C_y" : None,
                    "mu_y" : None, 
                    ### Ground truth source vectors. This part is only for debugging.
                    "Sgt" : S_wavelet,
                    "debug_iteration_point" : 10000,
                    "plot_debug_during_training" : False,
    }
    model = PredictiveDecorrBSS(**hyperparam_dict)
    model.fit(X_wavelet, shuffle_samples = True)

    Y_ = model.predict(X)
    Y_ = model.signed_and_permutation_corrected_sources(S, Y_) # Find sign and permutation ambiguity
    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
    Y_ = coef_ * Y_

    SINR_result = model.ComputeSINR(S, Y_)
    SNR_result = model.ComputeSNR(S, Y_)

    print("Signal-to-Interference-and-Noise-Ratio (SINR): {}".format(SINR_result))
    print("Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(SNR_result))

    result_dict_current = {
        'Model': 'PredictiveDecorrBSS',
        'seed': seed_,
        'SINR': SINR_result,
        'SNR': [SNR_result],
        'SNRinp': target_SNRinp,
        'execution_time': None
    }
    results_data.append(result_dict_current)
    result_df_current = pd.DataFrame(result_dict_current)
    RESULTS_DF = pd.DataFrame(results_data)
    RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))