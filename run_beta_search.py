
import os
from os.path import join, exists
from utils import load_splits, load_models, split_gen, parsers
from utils_model_sampling import hyperparameter_utils, sample_across_models
from utils_child import child_models

import configuration
config = configuration.Config()

import pandas as pd

import matplotlib.pyplot as plt
 
import numpy as np

import argparse
from datetime import datetime

def optimize_beta_and_lambda(split_name, dataset_name, model_dict, model_type):
 
    beta_sample = hyperparameter_utils.get_hyperparameter_search_values('beta')
    lambda_sample = hyperparameter_utils.get_hyperparameter_search_values('lambda')
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    
    this_exp_path = hyperparameter_utils.load_hyperparameter_folder(split_name, dataset_name, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type)
    
    if not exists(this_exp_path):
        os.makedirs(this_exp_path)
    
    # Calculated over all of CHILDES (data pool for all/all split).
    # Internally uses GPU if available.
    # speaker tags handled internally in the transformers bert completions file.
    
    success_utts_sample = load_splits.load_sample_successes(split_name, dataset_name).utterance_id
        
    # Don't use failures for beta search
    this_raw_beta_lambda_results = sample_across_models.sample_across_models(success_utts_sample,
                                                                      [], 
                                                                      model_dict,
                                                                      beta_sample, lambda_sample)
    
    this_raw_beta_results = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'levdist']
    this_raw_lambda_results = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'wfst']


    
    # Log the beta results
    this_beta_results_surp = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'levdist'].groupby(['beta_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    this_beta_results_surp = this_beta_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
    beta_results_path = join(this_exp_path, f'beta_search_results_{config.n_beta}.csv')
    this_beta_results_surp.to_csv(beta_results_path)
    print("Writing beta results to", {beta_results_path})
    plot_hyperparameter_optimization(this_exp_path, 'beta', beta_sample, this_beta_results_surp['posterior_surprisal'], split_name, dataset_name)
    
    
    # Log the lamba results
    this_lambda_results_surp = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'wfst'].groupby(['lambda_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    this_lambda_results_surp = this_lambda_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
    lambda_results_path = join(this_exp_path, f'lambda_search_results_{config.n_beta}.csv')
    this_lambda_results_surp.to_csv(lambda_results_path)
    print("Writing lambda results to", {lambda_results_path})
    plot_hyperparameter_optimization(this_exp_path, 'lambda', lambda_sample, this_lambda_results_surp['posterior_surprisal'], split_name, dataset_name)
    


    return this_raw_beta_results, this_beta_results_surp, this_raw_lambda_results, this_lambda_results_surp
    
def plot_hyperparameter_optimization(fig_path_dir, hyperparameter_name, hyperparameters, hyperparameter_surprisals, split, dataset):
    
    plt.title(hyperparameter_name +f' optimization for Split: {split}, Dataset: {dataset}')
    plt.xlabel(hyperparameter_name+' value')
    plt.ylabel('Posterior surprisal')
    plt.scatter(hyperparameters, hyperparameter_surprisals)
    
    fig_path = join(fig_path_dir, hyperparameter_name+f'_optimization_{config.n_beta}.png')
    plt.savefig(fname = fig_path)
    plt.close()
    print(f'Writing optimization plot to: {fig_path}')
    return fig_path
    
if __name__ == '__main__':    
    
    start_time = str(datetime.today())
    parser = parsers.split_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # end cite
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_model_args = vars(raw_args)
    
    query_model_str = load_models.get_model_id(
        split_name = this_model_args['split'],
        dataset_name = this_model_args['dataset'],
        with_tags =  this_model_args['use_tags'],
        context_width = this_model_args['context_width'],
        model_type = this_model_args['model_type']
    )
    
    print(this_model_args)
   
    if this_model_args['split'] != 'child':
        this_model_dict = load_models.get_model_dict(
            this_model_args['split'],
            this_model_args['dataset'],
            this_model_args['use_tags'],
            this_model_args['context_width'],
            this_model_args['model_type'],
        )
    else:
        
        this_model_dict = child_models.get_child_model_dict(this_model_args['dataset'])
        
        assert this_model_dict['kwargs']['use_speaker_labels'] == this_model_args['use_tags']
        assert this_model_dict['kwargs']['context_width_in_utts'] == this_model_args['context_width']
        
    hyperparameter_args = (this_model_args['split'], this_model_args['dataset'], this_model_dict, this_model_args['model_type'])
    raw_beta_results, beta_results, raw_lambda_results, lambda_results = optimize_beta_and_lambda(*hyperparameter_args)

    print(f'Computations complete for: {query_model_str}')
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')