
import os
from os.path import join, exists

from utils import load_models, load_splits, data_cleaning
from utils_model_sampling import beta_utils, sample_models_across_time

import numpy as np

import pandas as pd
import config

def load_sample_model_across_time_args(split_name, dataset_name):
    
    print('For child datasets, need to estimate beta based on all data available -- change this in the code.')
    
    eval_data_dict = load_splits.load_eval_data_all(split_name, dataset_name) 
    this_utts_with_ages = pd.concat([eval_data_dict['success_utts'], eval_data_dict['yyy_utts']])
    this_tokens_phono = eval_data_dict['phono']
        
    return this_utts_with_ages, this_tokens_phono


def call_single_across_time_model_unigram(unigram_name):
    
    assert unigram_name in {'data_unigram', 'flat_unigram'}
    return call_single_across_time_model(unigram_name, 'all', 'all', False, 0)

def call_single_across_time_model_bert(model_type, split, dataset, with_tags, context_num):
    
    name = load_models.get_model_id(split, dataset, with_tags, context_num, model_type)
    return call_single_across_time_model(model_type, split, dataset, with_tags, context_num)
    
    
def call_single_across_time_model(model_class, this_split, this_dataset_name, is_tags, context_width):
       
    assert model_class in {'childes', 'adult', 'flat_unigram', 'data_unigram'}, "Invalid model type presented."
   
    model_name = load_models.get_model_id(this_split, this_dataset_name, is_tags, context_width, model_class)
    
    all_models = load_models.get_model_dict()
    this_model_dict = all_models[model_name]
    
    utts, tokens = load_sample_model_across_time_args(this_split, this_dataset_name)
    
    utts = data_cleaning.get_target_child_year(utts)
    
    # Load the optimal beta
    optimal_beta = beta_utils.get_optimal_beta_value(this_split, this_dataset_name, this_model_dict, model_class)
    
    ages = np.unique(utts.year)
    
    for age in ages[:1]: # For development purposes only
    #for age in ages:
        
        this_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, utts, this_model_dict, tokens, beta_value = optimal_beta)
        
        if 'unigram' not in model_name:
            this_tags = this_model_dict['kwargs']['use_speaker_labels']
            this_context_width = this_model_dict['kwargs']['context_width_in_utts']
        else:
            this_tags = False; this_context_width = 0
            
        
        score_folder = load_beta_folder(this_split, this_dataset_name, this_tags, this_context_width, model_class)
        this_scores.to_csv(join(score_folder, 'run_models_across_time_{age}.csv'))# Need to assemble via model, then age later.
    
    
    
if __name__ == '__main__':
    
    #model_args = config.model_args
    
    # For now, run all models sequentially -- can refactor to argparse/parallel calls if needed, or if the model calls are too slow.
    # Need to parallelize this + run this all separately.
    # Test this for running on all_debug?
    
    # Need to test this on?
    
    # Run the unigrams
    #for unigram_name in ['flat_unigram', 'data_unigram']:
    #    # Only compute on all/all
    #    print(f'calling single across time model on {unigram_name}')
    #    call_single_across_time_model_unigram(unigram_name)

        
    # Run the adult BERT models
    #for context_width in config.context_list:
    #    call_single_across_time_model_bert('adult', 'all', 'all', False, context_width)
    
    
    # Run the CHILDES models
    
#     for split, dataset_name in [('all_debug', 'all_debug')]: # model_args:
#         for use_tags in [True, False]:
#             for context_num in config.context_list: # How to also call the unigram?
#                 print(f"calling single across time model {split}, {dataset_name}")
#                 call_single_across_time_model_bert('childes', split, dataset_name, use_tags, context_num)

    call_single_across_time_model_bert('childes', 'age', 'old', True, 0)
                
    # How to generate the bash scripts for this?
    