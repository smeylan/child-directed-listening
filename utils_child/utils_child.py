
import copy

from utils import load_splits, load_models, transformers_bert_completions, wfst
from utils_model_sampling import hyperparameter_utils
from utils_child import child_models, child_split_gen

import configuration
config = configuration.Config()

import os
from os.path import join, exists

import random
import pandas as pd 
import numpy as np


def load_cross_data(child_name):
    
    all_phono = load_splits.load_phono()
    child_phono = all_phono[all_phono.target_child_name == child_name]
    this_phono = child_phono[child_phono.phase_child_sample == config.eval_phase]
    
    return this_phono

def load_success_yyy_utts(data_type, child_name, display_all = False):
    
    cross_data = load_cross_data(child_name)
        
    if config.subsample_mode:
        this_attr = child_split_gen.get_subsample_key(config.n_used_score_subsample, data_type, child_name)
        data_to_extract = cross_data[cross_data[this_attr] == True]
    else:
        data_to_extract = cross_data
       
    utt_ids = sorted(list(set(data_to_extract.utterance_id)))
    return pd.DataFrame.from_records({'utterance_id' : utt_ids})
        

def load_success_utts(child_name = None, display_all = False):
    return load_success_yyy_utts('success', child_name, display_all)


def load_yyy_utts(child_name = None, display_all = False):
    return load_success_yyy_utts('yyy', child_name, display_all)

    
def get_cross_path(data_child_name, prior_child_name):
    
    this_folder = join(config.scores_dir, 'child_cross')
    
    if not exists(this_folder):
        os.makedirs(this_folder)
    
    this_path = join(this_folder, f'data_{data_child_name}_prior_{prior_child_name}.pkl')
    return this_path
    
def score_cross_prior(data_child, prior_child, likelihood_type):
    
    """
    Calculate one child's posterior distr. on their data
        with the prior of another child.
    """
    
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab = load_models.get_initial_vocab_info()
    _, is_tags = child_models.get_best_child_base_model_path()
    
    this_cross_data = load_cross_data(data_child)
    success_utts = load_success_utts(data_child).utterance_id
    yyy_utts = load_yyy_utts(data_child).utterance_id
    
    if likelihood_type == 'wfst':
        optim_hyperparam_val = hyperparameter_utils.get_optimal_hyperparameter_value('child', prior_child, is_tags, 0, 'childes', 'lambda')

        if config.fail_on_lambda_edge:
            not_at_edge = lambda this_lambda_sel : (abs(optim_hyperparam_val - config.lambda_high) < 1e-6) and (abs(optimal_lambda - config.lambda_low) < 1e-6)
            assert not_at_edge, f"Terminating, lambda value is {optimal_beta} which is too close to the edge."
                    
    elif likelihood_type == 'levdist':
        optim_hyperparam_val = hyperparameter_utils.get_optimal_hyperparameter_value('child', prior_child, is_tags, 0, 'childes', 'beta')
        if config.fail_on_beta_edge:
            not_at_edge = lambda this_beta_sel : (abs(optim_hyperparam_val - config.beta_high) < 1e-6) and (abs(optimal_beta - config.beta_low) < 1e-6)
            assert not_at_edge, f"Terminating, beta value is {optimal_beta} which is too close to the edge."



    if config.fail_on_lambda_edge:
        not_at_edge = lambda this_lambda_sel : (abs(optimal_lambda - config.lambda_high) < 1e-6) and (abs(optimal_lambda - config.lambda_low) < 1e-6)
        assert not_at_edge, f"Terminating, lambda value is {optimal_beta} which is too close to the edge."
    
    # Load the prior
    model = child_models.get_child_model_dict(prior_child)
    
    cross_priors = transformers_bert_completions.compare_successes_failures(this_cross_data, success_utts, yyy_utts, **model['kwargs'])
    
    # Calculate distances -- depending on how implementation is done hopefully can abstract this out.
    
    if likelihood_type == 'wfst': 
        likelihood_matrix, ipa = wfst.get_wfst_distance_matrix(this_cross_data, cross_priors, initial_vocab,  cmu_in_initial_vocab, config.fst_path, config.fst_sym_path)
        likelihood_matrix = -1 * np.log(likelihood_matrix + 10**-20)
    
    elif likelihood_type == 'levdist':
        likelihood_matrix = transformers_bert_completions.get_edit_distance_matrix(this_cross_data, 
            cross_priors, cmu_in_initial_vocab)
    else:
        assert False, "Invalid dist specified in config file. Choose from: {levdist}"


    # in either case, reduce the dimensionality of the likelihood matrix to find the best pronunciation in each case
    likelihood_matrix = wfst.reduce_duplicates(likelihood_matrix, cmu_in_initial_vocab, initial_vocab, 'min', cmu_indices_for_initial_vocab)
    
    if likelihood_type == 'wfst': 
        posteriors = transformers_bert_completions.get_posteriors(cross_priors, 
                    likelihood_matrix, initial_vocab, None, optim_hyperparam_val)

    elif likelihood_type == 'levdist':
        posteriors = transformers_bert_completions.get_posteriors(cross_priors, 
                    likelihood_matrix, initial_vocab, None, optim_hyperparam_val)        
    
    
    posteriors['scores']['optim_hyperparam_val'] = optim_hyperparam_val
    posteriors['scores']['likelihood_type'] = likelihood_type
    posteriors['scores']['model'] = model['title']
    scores = copy.deepcopy(posteriors['scores'])
    
    return scores, optim_hyperparam_val

    
