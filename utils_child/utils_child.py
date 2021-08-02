
import copy

from utils import load_splits, load_models, transformers_bert_completions
from utils_model_sampling import beta_utils
from utils_child import child_models

import config

import os
from os.path import join, exists

import random
import pandas as pd 

def load_cross_data(child_name, all_phono = None):
    
    all_phono = load_splits.load_phono()
    child_phono = all_phono[all_phono.target_child_name == child_name]
    this_phono = child_phono[child_phono.phase_child_sample == config.eval_phase]
    
    return this_phono

def load_success_yyy_utts(data_type, child_name, cross_data):
    
    
    if not isinstance(child_name, str):
        cross_data = child_name # Didn't specify positional argument.
        child_name = None
   
    assert (child_name is None) ^ (cross_data is None), "Specify one of either child_name or cross_data." 
    
    if cross_data is None:
        cross_data = load_cross_data(child_name)
    
    # For now: Possibly non-reproducible sampling from the right phase (intermediate results), or, seed the data and see if it results in a reproducible split.
    
    utt_ids = random.shuffle(list(set(cross_data[cross_data.partition == data_type].utterance_id)))
    
    if config.dev_mode:
        utt_ids = utt_ids[:min(len(utt_ids), config.n_subsample)]
     
    return pd.DataFrame.from_records({'utterance_id' : utt_ids})
        

def load_success_utts(child_name = None, cross_data = None):
    return load_success_yyy_utts('success', child_name, cross_data)


def load_yyy_utts(child_name = None, cross_data = None):
    return load_success_yyy_utts('yyy', child_name, cross_data)

    
def get_cross_path(data_child_name, prior_child_name, beta):
    
    this_folder = join(config.exp_dir, 'child_cross')
    
    if not exists(this_path):
        os.makedirs(this_folder)
    
    this_path = join(this_folder, f'data_{data_child_name}_prior_{prior_child_name}_beta_{beta}.pkl')
    return this_path
    
def score_cross_prior(data_child, prior_child):
    
    """
    Calculate one child's posterior distr. on their data
        with the prior of another child.
    """
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    _, is_tags = child_models.get_best_child_base_model_path()
    
    print('Re-enable optimal beta for child once these values become available!')
    
    this_cross_data = load_cross_data(data_child)
    success_utts = load_success_utts(this_cross_data).utterance_id
    yyy_utts = load_yyy_utts(this_cross_data).utterance_id
    
    # optim_beta = beta_utils.get_optimal_beta_value('child', prior_child, is_tags, 0, 'childes')
    optim_beta = 3.2
    
    # Load the evaluation successes and yyy for a given child.
    eval_data = load_splits.load_pvd_data('child', data_child, config.eval_phase)
    
    # Load the prior
    model = child_models.get_child_model_dict(prior_child)
    
    # Use id, not utterance id, because this is Providence second query data.
    cross_priors = transformers_bert_completions.compare_successes_failures(this_cross_data, success_utts, yyy_utts, **model['kwargs'])
    
    # Calculate distances -- depending on how implementation is done hopefully can abstract this out.
    
    dists = None
    
    if config.dist_type == 'levdist':
        dists = transformers_bert_completions.get_edit_distance_matrix(eval_data['phono'], 
            cross_priors, initial_vocab, cmu_in_initial_vocab)    
    else:
        assert False, "Invalid dist specified in config file. Choose from: {levdist}"
    
    posteriors_for_age_interval = transformers_bert_completions.get_posteriors(cross_priors, 
                    dists, initial_vocab, None, optim_beta)
    
    posteriors_for_age_interval['scores']['beta_value'] = optim_beta
    posteriors_for_age_interval['scores']['model'] = model['title']
        
    scores = copy.deepcopy(posteriors['scores'])
    
    return scores, optim_beta
    