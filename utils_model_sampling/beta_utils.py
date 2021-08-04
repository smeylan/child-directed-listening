
import os
from os.path import join, exists

import numpy as np

from utils import load_models, split_gen
import config

import pandas as pd

def get_beta_search_values():
    
    low = config.beta_low
    high = config.beta_high
    num_values = config.num_values
    
    beta_samples = np.arange(low, high, (high - low) / num_values)
    
    return beta_samples


def load_beta_folder(split, dataset, is_tags, context_num, model_type):
    
    folder = split_gen.get_split_folder(split, dataset, config.scores_dir)
    this_title = load_models.query_model_title(split, dataset, is_tags, context_num, model_type)
    exp_path = join(folder, this_title.replace(' ', '_'))
    
    if not exists(exp_path):
        os.makedirs(exp_path)
    
    return exp_path

def load_beta_values(split_name, dataset_name, tags, context_width, model_type):
    
    exp_model_path = load_beta_folder(split_name, dataset_name, tags, context_width, model_type)
    results = pd.read_csv(join(exp_model_path, f'beta_search_results_{config.n_beta}.csv'))

    return results


def get_optimal_beta_value_with_dict(split, dataset, model_dict, model_type):
    
    return get_optimal_beta_value(split, dataset, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type)
    
    
def get_optimal_beta_value(split, dataset, tags, context, model_type):
    
    this_beta_results  = load_beta_values(split, dataset, tags, context, model_type)
    
    # Need to argmax for beta_value, given the posterior surprisal
    list_beta_results = list(this_beta_results['beta_value'])
    list_surp = list(this_beta_results['posterior_surprisal'])
    
    argmin_beta = np.argmin(list_surp)
    best_beta = list_beta_results[argmin_beta]

    return best_beta
    
    
    
    
    
