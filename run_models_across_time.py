
import os
from os.path import join, exists

from utils import load_models, load_splits, data_cleaning, parsers
from utils_model_sampling import beta_utils, sample_models_across_time

import numpy as np

import pandas as pd
import config

from collections import defaultdict

from datetime import datetime
    
    
def call_single_across_time_model(sample_dict, all_tokens_phono, model_class, this_split, this_dataset_name, is_tags, context_width):
       
    assert model_class in {'childes', 'adult', 'flat_unigram', 'data_unigram'}, "Invalid model type presented."
   
    model_name = load_models.get_model_id(this_split, this_dataset_name, is_tags, context_width, model_class)
    
    this_model_dict = load_models.get_model_dict(
        this_split,
        this_dataset_name,
        is_tags,
        context_width,
        model_class,
    )
         
    # Load the optimal beta
    optimal_beta = beta_utils.get_optimal_beta_value_with_dict(this_split, this_dataset_name, this_model_dict, model_class)
    
    ages = sorted(list(sample_dict.keys()))
   
    for idx, age_str in enumerate(ages):
        
        print('Processing', model_class, age_str) 
        
        age = float(age_str)
        
        percentage_done = idx / float(len(ages)) * 100
        
        if int(percentage_done) % 10 == 0: print(f'{percentage_done}%')
            
        this_pool = sample_dict[age_str]
        this_success_pool = this_pool['success']
        this_yyy_pool = this_pool['yyy']
        
        if (this_success_pool.shape[0] == 0) and (this_yyy_pool.shape[0] == 0): continue
         
        this_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, this_model_dict, all_tokens_phono, optimal_beta)
            
        this_tags = this_model_dict['kwargs']['use_speaker_labels']
        this_context_width = this_model_dict['kwargs']['context_width_in_utts']
        
        score_folder = beta_utils.load_beta_folder(this_split, this_dataset_name, this_tags, this_context_width, model_class)
        
        this_scores.to_pickle(join(score_folder, f'run_models_across_time_{age_str}.pkl'))
    
    return this_scores
    
if __name__ == '__main__':
    
    start_time = str(datetime.today())
    
    parser = parsers.split_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_model_args = vars(raw_args)
    
    query_model_str = load_models.get_model_id(
        split_name = this_model_args['split'],
        dataset_name = this_model_args['dataset'],
        with_tags =  this_model_args['use_tags'],
        context_width = this_model_args['context_width'],
        model_type = this_model_args['model_type'],
    )    

    print('Need to update this for children.')
                                                                    
    all_phono = load_splits.load_phono()
    this_sample_dict = load_splits.load_sample_model_across_time_args()
     
    scores = call_single_across_time_model(this_sample_dict, all_phono, this_model_args['model_type'], this_model_args['split'], this_model_args['dataset'], this_model_args['use_tags'], this_model_args['context_width'])
    
    print(f'Computations complete for: {query_model_str}')
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
    
    
    