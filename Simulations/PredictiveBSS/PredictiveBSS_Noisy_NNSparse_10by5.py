import os
import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd

from bss.bss_utils import generate_uncorrelated_uniform_sources, addWGN, ProjectRowstoL1NormBall
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS
from bss.LDMIBSS import LDMIBSS
from bss.CorInfoMaxBSS import OnlineCorInfomax
from python_utils.python_utils import Timer

if not os.path.exists("../Results"):
    os.mkdir("../Results")

# RESULTS_DF = pd.DataFrame( columns = ['seed', 'Model', 'SINR', 'SNR', 'SNRinp', 'execution_time'])
pickle_name_for_results = "predictive_bss_noisy_nnsparse_10by5_results.pkl"
### Setting of the simulation and the model hyperparameters.
N = 100000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

predictivebss_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "nnsparse",
                "epsilon" : 1e-4,
                ### Optimization parameters
                "lambda_lateral" : 0.99,
                "gamma_predictive" : 250,
                ### Learning rates 
                "lr_W" : 5 * 1e-2,
                "neural_lr_start" : 0.1,
                "neural_lr_stop" : 1e-4,
                "stlambda_lr" : 0.5,
                "neural_dynamics_iterations" : 100,
                "neural_OUTPUT_COMP_TOL" : 1e-7,
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
                "debug_iteration_point" : 25000,
                "plot_debug_during_training" : False,
}

online_corinfomax_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "nnsparse",
                ### Optimization parameters
                "lambda_lateral" : 0.99,
                "gamma_predictive" : 5,
                ### Learning rates 
                "lr_W" : 1 * 1e-1,
                "neural_lr_start" : 0.1,
                "neural_lr_stop" : 1e-3,
                "stlambda_lr" : 0.1,
                "neural_dynamics_iterations" : 250,
                "neural_OUTPUT_COMP_TOL" : 1e-7,
                ### Learning rate rules and decay parameters
                "lr_W_rule" : "divide_by_log_index",
                "lr_W_decay_divider" : 5000,
                "neural_lr_rule" : "divide_by_loop_index",
                "neural_lr_decay_divider" : 200,
                ### Initial values for weights if provided, if not they will be initialized in the fit function 
                "W" : None,
                "B_y" : None,
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 10000,
                "plot_debug_during_training" : False,
}
ldmi_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "nnsparse",
                "method" : "covariance",
                ### Optimization parameters and Learning rates 
                "mu_y_start" : 100,
                "mu_y_rule" : "divide_by_root_index",
                "epsilon" : 1e-3,
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 1000,
                "plot_debug_during_training" : False,
}

input_snr_list = [30., 25., 20., 15., 10., 5.]
seed_list = np.arange(1, 3001, 100)
results_data = []
for snr_ in input_snr_list:
    for seed in seed_list:
        np.random.seed(seed)
        print("seed is ", seed)
        ##################################################
        ##### GENERATE SOURCES ###########################
        ##################################################
        S = generate_uncorrelated_uniform_sources(NumberofSources, N, min_val = -2, max_val = 2)
        S = ProjectRowstoL1NormBall(S.T).T 
        S = S * (S >= 0)

        A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
        X_noNoise = np.dot(A, S)

        target_SNRinp = snr_
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X)**2, axis=1))
        )

        ##################################################
        ############ Predictive Decor BSS ################
        ##################################################
        print("Running PredictiveDecorr Model")
        with Timer() as t:
            model = PredictiveDecorrBSS(**predictivebss_hyperparam_dict, Sgt = S)
            model.C_y = np.eye(NumberofSources) / 10 + np.random.randn(NumberofSources, NumberofSources) / 15
            model.W = np.random.randn(NumberofSources, NumberofMixtures) / 15 + np.eye(NumberofSources, NumberofMixtures) * 0.01
            model.fit(X)

        # In[7]:

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
            'seed': seed,
            'SINR': SINR_result,
            'SNR': [SNR_result],
            'SNRinp': target_SNRinp,
            'execution_time': t.interval
        }
        results_data.append(result_dict_current)
        result_df_current = pd.DataFrame(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))


        ##################################################
        ############ CorInfoMax BSS ######################
        ##################################################
        print("Running CorInfoMax Model")
        with Timer() as t:
            model = OnlineCorInfomax(**online_corinfomax_hyperparam_dict, Sgt = S)
            model.fit(X)

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_) # Find sign and permutation ambiguity
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
        Y_ = coef_ * Y_

        SINR_result = model.ComputeSINR(S, Y_)
        SNR_result = model.ComputeSNR(S, Y_)
        print("Signal-to-Interference-and-Noise-Ratio (SINR): {}".format(SINR_result))
        print("Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(SNR_result))

        result_dict_current = {
            'Model': 'CorInfoMax',
            'seed': seed,
            'SINR': SINR_result,
            'SNR': [SNR_result],
            'SNRinp': target_SNRinp,
            'execution_time': t.interval
        }
        results_data.append(result_dict_current)
        result_df_current = pd.DataFrame(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

        ##################################################
        ############## LDMIBSS ###########################
        ##################################################
        print("Running LDMIBSS Model")
        with Timer() as t:
            model = LDMIBSS(**ldmi_hyperparam_dict,
                            Sgt = S[:, :10000])
            model.fit(X[:, :10000], regularize_W = True)

        Y_ = model.predict(X)
        Y_ = model.signed_and_permutation_corrected_sources(S, Y_) # Find sign and permutation ambiguity
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
        Y_ = coef_ * Y_

        SINR_result = model.ComputeSINR(S, Y_)
        SNR_result = model.ComputeSNR(S, Y_)
        print("Signal-to-Interference-and-Noise-Ratio (SINR): {}".format(SINR_result))
        print("Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(SNR_result))

        result_dict_current = {
            'Model': 'LDMIBSS',
            'seed': seed,
            'SINR': SINR_result,
            'SNR': [SNR_result],
            'SNRinp': target_SNRinp,
            'execution_time': t.interval
        }
        results_data.append(result_dict_current)
        result_df_current = pd.DataFrame(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))
