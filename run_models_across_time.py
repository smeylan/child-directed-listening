
import os
from os.path import join, exists

from utils import load_models, load_splits
from sample_models_across_time import successes_across_time_per_model
from utils_model_sampling import beta_utils

import pandas as pd
import config

def load_sample_model_across_time_args(split_name, dataset_name):
    
    print('For child datasets, need to estimate beta based on all data available -- change this in the code.')
    
    this_utts_save_path = join(config.eval_dir, model_name)
        
    eval_data_dict = load_splits.load_eval_data_all(split_name, dataset_name) 
    this_utts_with_ages = pd.concat([eval_data_dict['success_utts'], eval_data_dict['yyy_utts']])
    this_tokens_phono = eval_data_dict['phono']
        
    return this_utts_with_ages, this_tokens_phono

def call_single_across_time_model(this_split, this_dataset_name, is_tags, context_width):
    """
    model_name is of the form: {split name}/{dataset name}
    """
    
    model_name = get_model_id(this_split, this_dataset_name, is_tags, context_width)
    
    all_models = load_models.get_model_dict()
    this_model_dict = all_models[model_name]
    
    utts, tokens = load_sample_model_across_time_args(this_split, this_dataset_name)
    
    # Load the optimal beta
    optimal_beta = beta_utils.get_optimal_beta_value(this_split, this_dataset_name, this_model_dict)
    
    for age in ages: # For which ages?
        # Need to think about tags/ no tags
        this_scores = successes_across_time_per_model(age, utts, this_model_dict, tokens, root_dir, beta_value = optimal_beta)
        
        beta_folder = load_beta_folder(this_split, this_dataset_name, this_model_dict['kwargs']['use_speaker_labels'], this_model_dict['kwargs']['context_width_in_utts'])
        
        this_scores.to_csv(join(beta_folder, 'run_models_across_time_{age}.csv')) # Need to assemble via model, then age later.
    
if __name__ == '__main__':
    
    model_args = [('all_debug', 'all_debug')]
    
    # For now, run all models sequentially -- can refactor to argparse/parallel calls if needed, or if the model calls are too slow.
    
    for split, dataset_name in model_args:
        for use_tags in [True, False]:
            for context_num in config.context_list:
                call_single_across_time_model(split, dataset_name, use_tags, context_num)
    