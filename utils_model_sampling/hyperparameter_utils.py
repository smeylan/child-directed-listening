
import os
from os.path import join, exists

import numpy as np

from utils import load_models, split_gen
import configuration
config = configuration.Config()

import pandas as pd

def get_hyperparameter_search_values(hyperparam):
    
    low = getattr(config, hyperparam+'_low')
    high = getattr(config, hyperparam+'_high')
    num_values = getattr(config, hyperparam+'_num_values')
    
    hyperparameter_samples = np.arange(low, high, (high - low) / num_values)
    
    return hyperparameter_samples


def load_hyperparameter_folder(split, dataset, is_tags, context_num, model_type):
    
    folder = split_gen.get_split_folder(split, dataset, config.scores_dir)
    this_title = load_models.query_model_title(split, dataset, is_tags, context_num, model_type)
    exp_path = join(folder, this_title.replace(' ', '_'))
    
    if not exists(exp_path):
        os.makedirs(exp_path)
    
    return exp_path

def load_hyperparameter_values(split_name, dataset_name, tags, context_width, model_type, hyperparameter):
    
    exp_model_path = load_hyperparameter_folder(split_name, dataset_name, tags, context_width, model_type)
    results = pd.read_csv(join(exp_model_path, hyperparameter+f'_search_results_{config.n_beta}.csv'))

    return results


def get_optimal_hyperparameter_value_with_dict(split, dataset, model_dict, model_type, hyperparameter):
    
    return get_optimal_hyperparameter_value(split, dataset, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type, hyperparameter)
    
    
def get_optimal_hyperparameter_value(split, dataset, tags, context, model_type, hyperparameter):
    
    this_hyperparameter_results  = load_hyperparameter_values(split, dataset, tags, context, model_type, hyperparameter)
    
    # Need to argmax for beta_value, given the posterior surprisal
    list_hyperparameter_results = list(this_hyperparameter_results[hyperparameter+'_value'])
    list_surp = list(this_hyperparameter_results['posterior_surprisal'])
    
    argmin_hyperparameter = np.argmin(list_surp)
    best_hyperparameter = list_hyperparameter_results[argmin_hyperparameter]

    return best_hyperparameter
    
    
    
    
    
