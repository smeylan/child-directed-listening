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
from src.utils import load_splits, load_models, split_gen, parsers, hyperparameter_utils, sample_across_models, child_models, configuration
config = configuration.Config()

def optimize_beta_and_lambda(split_name, dataset_name, model_dict, model_type, training_dataset_name, training_split_name):
    '''
        Find the values of beta and lambda which minimize posterior surprisal; save this information in a place that run_models_across_time can load

        Args:
        split_name: If the model is a fine-tuned BERT model, is it trained on all CHILDES data, young children, or old chilren
        dataset_name: what dataset should be evaluated?
        model_dict: A model dictionary from the load models functions (not a HuggingFace model alone!)
        model_type: model label, choose 'childes' for fine-tuned BERT, 'adult' for off the shelf BERT, 'flat_unigram' for UniformPrior, 'data_unigram' for CHILDES-unigram
        
        Return: the best parameter values for WFST and Levenshtein distance likelihoods and accompanying scores. Plots these results as a side effect.

    '''

    beta_sample = hyperparameter_utils.get_hyperparameter_search_values('beta')
    lambda_sample = hyperparameter_utils.get_hyperparameter_search_values('lambda')
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab  = load_models.get_initial_vocab_info()
    
    # this needs to be distunguished betweeen a test and a training dataset
    this_exp_path = hyperparameter_utils.load_hyperparameter_folder(training_split_name, dataset_name, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type, training_dataset_name)
    
    if not exists(this_exp_path):
        os.makedirs(this_exp_path)
    
    # use the split_name, not the training_split_name to determine the test set
    success_utts_sample = load_splits.load_sample_successes(split_name, dataset_name).utterance_id
        
    # Don't use failures for beta search
    this_raw_beta_lambda_results = sample_across_models.sample_across_models(success_utts_sample, [], model_dict, beta_sample, lambda_sample)
    
    this_raw_beta_results = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'levdist']
    this_raw_lambda_results = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'wfst']

    # Log the beta results
    this_beta_results_surp = this_raw_beta_lambda_results.loc[this_raw_beta_lambda_results.likelihood_type == 'levdist'].groupby(['beta_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))).reset_index()
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

    '''
    Generate figures to look at scores across each hyperparamter range

    Args:
    fig_path_dir: directory to output to
    hyperparameter_name: 'beta' or 'lambda'
    hyperparameters: values of the hyperparameters (x axis)
    hyperparameter_surprisals: scores associated with each hyperparameter
    split: which subset of samples should be used to compute the scores
    dataset: which dataset should this be scored against 

    Return:
    Path to the figure saved to disk

    '''
    
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
    this_model_args = vars(raw_args)
    
    # If training_dataset or training_split is defined, use its value to determine the model to load. Otherwise assume that `dataset` and `split` are overloaded and that the value should be used in order to choose both the dataet to test against and the model to load

    if not this_model_args['training_dataset']:
        this_model_args['training_dataset'] = this_model_args['dataset']

    if not this_model_args['training_split']:
        this_model_args['training_split'] = this_model_args['split']

    query_model_str = load_models.get_model_id(
        split_name = this_model_args['training_split'],
        dataset_name = this_model_args['training_dataset'],
        use_tags =  this_model_args['use_tags'],
        context_width = this_model_args['context_width'],
        model_type = this_model_args['model_type']
    )
    
    print(this_model_args)
   
    if this_model_args['split'] != 'child':
        this_model_dict = load_models.get_model_dict(this_model_args['training_split'], this_model_args['training_dataset'], this_model_args['use_tags'],this_model_args['context_width'],this_model_args['model_type'],)
    else:
        
        this_model_dict = child_models.get_child_model_dict(this_model_args)
        
        assert this_model_dict['kwargs']['use_speaker_labels'] == this_model_args['use_tags']
        assert this_model_dict['kwargs']['context_width_in_utts'] == this_model_args['context_width']

    # second argument is the test dataset; if the training dataset has been altered, it's in the model_dict that gets passed
    hyperparameter_args = (this_model_args['split'], this_model_args['dataset'], this_model_dict, this_model_args['model_type'], this_model_args['training_dataset'], this_model_args['training_split'])
    raw_beta_results, beta_results, raw_lambda_results, lambda_results = optimize_beta_and_lambda(*hyperparameter_args)

    print(f'Computations complete for: {query_model_str}')
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')