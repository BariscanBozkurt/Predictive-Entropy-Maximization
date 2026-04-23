import os
import sys
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
from bss.NSMBSS import OnlineNSM
from python_utils.python_utils import Timer

print("Running script PredictiveBSS_Correlated_NNAntisparse_10by5")
if not os.path.exists("../Results"):
    os.mkdir("../Results")

# RESULTS_DF = pd.DataFrame( columns = ['seed', 'Model', 'SINR', 'SNR', 'SNRinp', 'execution_time'])
pickle_name_for_results = "predictive_bss_correlated_nnantisparse_10by5_results.pkl"
### Setting of the simulation and the model hyperparameters.
N = 100000
NumberofSources = 5
NumberofMixtures = NumberofSources + 5

predictivebss_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "nnantisparse",
                ### Optimization parameters
                "lambda_lateral" : 0.95,
                "gamma_predictive" : 750,
                ### Learning rates 
                "lr_W" : 5 * 1e-2,
                "neural_lr_start" : 0.05,
                "neural_lr_stop" : 1e-4,
                "neural_dynamics_iterations" : 500,
                "neural_OUTPUT_COMP_TOL" : 1e-6,
                ### Learning rate rules and decay parameters
                "lr_W_rule" : "divide_by_index",
                "lr_W_decay_divider" : 20000,
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
                "presumed_domain" : "nnantisparse",
                ### Optimization parameters
                "lambda_lateral" : 0.99,
                "gamma_predictive" : 15,
                ### Learning rates 
                "lr_W" : 10 * 1e-2,
                "neural_lr_start" : 0.9,
                "neural_lr_stop" : 1e-6,
                "neural_dynamics_iterations" : 500,
                "neural_OUTPUT_COMP_TOL" : 1e-6,
                ### Learning rate rules and decay parameters
                "lr_W_rule" : "divide_by_log_index",
                "lr_W_decay_divider" : 10000,
                "neural_lr_rule" : "divide_by_loop_index",
                "neural_lr_decay_divider" : 500,
                ### Initial values for weights if provided, if not they will be initialized in the fit function 
                "W" : None,
                "B_y" : 5*np.eye(NumberofSources),
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 10000,
                "plot_debug_during_training" : True,
}

online_nsm_hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "neural_dynamics_iterations" : 250,
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 10000,
                "plot_debug_during_training" : True,
}

ldmi_hyperparam_dict = hyperparam_dict = {
                "n_sources" :  NumberofSources,
                "presumed_domain" : "nnantisparse",
                "method" : "covariance",
                ### Optimization parameters and Learning rates 
                "mu_y_start" : 100,
                "mu_y_rule" : "exponential_decay",
                "epsilon" : 1e-5,
                ### Ground truth source vectors. This part is only for debugging.
                "debug_iteration_point" : 1000,
                "plot_debug_during_training" : False,
}

rho_list = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
seed_list = np.arange(1, 3001, 100)
results_data = []
for rho in rho_list:
    for seed in seed_list:
        np.random.seed(seed)
        print("seed is ", seed)
        ##################################################
        ##### GENERATE SOURCES ###########################
        ##################################################
        # # Generate Sources and Mix Them (10 by 5 case)
        S = generate_correlated_copula_sources( rho=rho, df=4,
                                                n_sources=NumberofSources, 
                                                size_sources=N, 
                                                decreasing_correlation=False)

        A = np.random.randn(NumberofMixtures, NumberofSources) # Random Gaussian mixing matrix
        X_noNoise = np.dot(A, S)

        target_SNRinp = 30
        X = addWGN(X_noNoise, target_SNRinp)

        SNRinp = 10 * np.log10(
            np.sum(np.mean(X_noNoise ** 2, axis=1))
            / np.sum(np.mean((X_noNoise - X)**2, axis=1))
        )

        ##################################################
        ####### PREDICTIVE BSS ###########################
        ##################################################
        print("Running Predictive BSS Model")
        with Timer() as t:
            model = PredictiveDecorrBSS(**predictivebss_hyperparam_dict,
                                        Sgt = S)
            model.C_y = np.eye(NumberofSources) * 2
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
            'rho' : rho,
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
        print("Running CorInfoMax BSS Model")
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
            'Model': 'CorInfoMaxBSS',
            'seed': seed,
            'rho' : rho,
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
        ################## ICA ###########################
        ##################################################
        print("Running ICA-InfoMax Model")
        with Timer() as t:
            Y_ = fit_icainfomax(X, 5)
        Y_ = BSSBaseClass().signed_and_permutation_corrected_sources(S - S.mean(1, keepdims = True), Y_) # Find sign and permutation ambiguity
        Y_ -= Y_.min(1, keepdims = True) # ICA returns zero mean predictions. Since we know the sources are positive, we make the estimated sources positive by adding a constant to them. 
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1) # Find if the extracted signals need some amplification! The networks learned weight may need amplification due to lateral connections during the neural dynamics!
        Y_ = coef_ * Y_

        SINR_result = BSSBaseClass().ComputeSINR(S, Y_)
        SNR_result = BSSBaseClass().ComputeSNR(S, Y_)
        print("Signal-to-Interference-and-Noise-Ratio (SINR): {}".format(SINR_result))
        print("Component Signal-to-Noise-Ratio (SNR) Values : {}\n".format(SNR_result))

        result_dict_current = {
            'Model': 'ICA_InfoMax',
            'seed': seed,
            'rho' : rho,
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
            'rho' : rho,
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
        ############### NSMBSS ###########################
        ##################################################
        print("Running NSM BSS Model")
        with Timer() as t:
            model = OnlineNSM(**online_nsm_hyperparam_dict, Sgt = S)
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
            'Model': 'NSMBSS',
            'seed': seed,
            'rho' : rho,
            'SINR': SINR_result,
            'SNR': [SNR_result],
            'SNRinp': target_SNRinp,
            'execution_time': t.interval
        }
        results_data.append(result_dict_current)
        result_df_current = pd.DataFrame(result_dict_current)
        RESULTS_DF = pd.DataFrame(results_data)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))