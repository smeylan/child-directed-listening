import os
from os.path import join, exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import sys

sys.path.append('.')
sys.path.append('src/.')
from src.utils import load_splits, load_models, split_gen, parsers, hyperparameter_utils, sample_across_models, child_models, configuration, paths
config = configuration.Config()

def fit_child_specific_wfst(fitting_dict):
    '''
        Determines whether a child-specific FST should be evaluated for this particular fitting_dict

        Args:
        fitting_dict: a dictionary with keys for training_split, training_dataset, test_split, test_dataset, etc. See utils/paths.py for a full description

        Return: Boolean as to whether the model should use a child-specific wfst

    '''


    # we should only get the results of the child-specific WFST when the child and the training data are the same
    return((fitting_dict['training_split'] == 'Providence-Child') and (fitting_dict['training_split'] == fitting_dict['test_split']))

def optimize_beta_and_lambda(fitting_dict):
    '''
        Find the values of beta and lambda which minimize posterior surprisal; save this information in a place that run_models_across_time can load

        Args:
        fitting dict: a dictionary with keys for training_split, training_dataset, test_split, test_dataset, etc. See utils/paths.py for a full description
        
        Return: the best parameter values for WFST and Levenshtein distance likelihoods; saves the scores for each hyperparameter value as a side effect

    '''

    beta_sample = hyperparameter_utils.get_hyperparameter_search_values('beta')
    lambda_sample = hyperparameter_utils.get_hyperparameter_search_values('lambda')
    if fit_child_specific_wfst(fitting_dict): # fit a parameter for the child-specific FST iff it's a fine-tuned child model
        gamma_sample = hyperparameter_utils.get_hyperparameter_search_values('lambda')
    else:
        gamma_sample = None

        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab  = load_models.get_initial_vocab_info()
    fitting_path =  paths.get_directory(fitting_dict)    
    
    if not exists(fitting_path):
        os.makedirs(fitting_path)
    
    success_utts_sample_path = paths.get_sample_csv_path(task_phase_to_sample_for='fit', split=fitting_dict['test_split'], dataset=fitting_dict['test_dataset'], data_type='success', age = None, n=config.n_beta)
    success_utts_sample  = pd.read_csv(success_utts_sample_path).utterance_id
        
    # Don't use failures for beta search
    if fit_child_specific_wfst(fitting_dict):
        hyperparam_search_results = sample_across_models.sample_across_models(success_utts_sample, [], fitting_dict, beta_sample, lambda_sample, gamma_sample, child_name = fitting_dict['training_dataset'])
    else:
        hyperparam_search_results = sample_across_models.sample_across_models(success_utts_sample, [], fitting_dict, beta_sample, lambda_sample, gamma_sample)

    
    this_raw_beta_results = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'levdist']
    this_raw_lambda_results = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'wfst']
    if (fitting_dict['training_split'] == 'Providence-Child') and (fitting_dict['training_split'] == fitting_dict['test_split']):
        this_raw_gamma_results = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'wfst-child']
    else:
        this_raw_gamma_results = None


    # Log the beta results
    this_beta_results_surp = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'levdist'].groupby(['beta_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))).reset_index()
    this_beta_results_surp = this_beta_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
    beta_results_path = join(fitting_path, f'beta_search_results_{config.n_beta}.csv')
    this_beta_results_surp.to_csv(beta_results_path)
    print("Writing beta results to", {beta_results_path})
    #plot_hyperparameter_optimization(fitting_path, 'beta', beta_sample, this_beta_results_surp['posterior_surprisal'], split_name, dataset_name)
    
    
    # Log the lamba results
    this_lambda_results_surp = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'wfst'].groupby(['lambda_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    this_lambda_results_surp = this_lambda_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
    lambda_results_path = join(fitting_path, f'lambda_search_results_{config.n_beta}.csv')
    this_lambda_results_surp.to_csv(lambda_results_path)
    print("Writing lambda results to", {lambda_results_path})
    #plot_hyperparameter_optimization(fitting_path, 'lambda', lambda_sample, this_lambda_results_surp['posterior_surprisal'], split_name, dataset_name)
    
    # log the gamma results if necessary
    if fit_child_specific_wfst(fitting_dict):
        this_gamma_results_surp = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'wfst-child'].groupby(['gamma_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))).reset_index()
        this_gamma_results_surp = this_gamma_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
        gamma_results_path = join(fitting_path, f'gamma_search_results_{config.n_beta}.csv')
        this_gamma_results_surp.to_csv(gamma_results_path)
        print("Writing gamma results to", {gamma_results_path})
    else:
        this_gamma_results_surp = None


    return this_raw_beta_results, this_beta_results_surp, this_raw_lambda_results, this_lambda_results_surp, this_raw_gamma_results, this_gamma_results_surp    
    
if __name__ == '__main__':    
    
    start_time = str(datetime.today())
    parser = parsers.split_parser()
        
    raw_args = parser.parse_known_args()[0]    
    this_model_args = vars(raw_args)

    this_model_args['task_phase'] = 'fit'
    this_model_args['n_samples'] = config.n_across_time   
    print(this_model_args)             
    
    this_model_dict = load_models.get_fitted_model_dict(this_model_args)

    print('Loaded the model!')    
    this_raw_beta_results, this_beta_results_surp, this_raw_lambda_results, this_lambda_results_surp, this_raw_gamma_results, this_gamma_results_surp = optimize_beta_and_lambda(this_model_dict)

    print(f'Computations complete for model:')
    print(this_model_dict)
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')