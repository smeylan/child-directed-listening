import os
import sys
from os.path import join, exists
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.append('.')
sys.path.append('src/.')
from utils import load_models, load_splits, data_cleaning, parsers, hyperparameter_utils, sample_models_across_time, configuration
config = configuration.Config()
  
    
def call_single_across_time_model(sample_dict, all_tokens_phono, model_class, this_split, this_dataset_name, is_tags, context_width, examples_mode):
       
    assert model_class in {'childes', 'adult', 'flat_unigram', 'data_unigram'}, "Invalid model type presented."
   
    model_name = load_models.get_model_id(this_split, this_dataset_name, is_tags, context_width, model_class)
    
    beta_model_dict = load_models.get_model_dict(
        this_split,
        this_dataset_name,
        is_tags,
        context_width,
        model_class,
    )

    lambda_model_dict = load_models.get_model_dict(
        this_split,
        this_dataset_name,
        is_tags,
        context_width,
        model_class,
    )
         
    # Load the optimal beta and lambda
    optimal_lambda_value = [hyperparameter_utils.get_optimal_hyperparameter_value_with_dict(this_split, this_dataset_name, lambda_model_dict, model_class, 'lambda')]
    if config.fail_on_lambda_edge:
        if optimal_lambda_value[0] >= config.lambda_high:
            raise ValueError('Lambda value is too high; examine the range for WFST scaling.')
        if optimal_lambda_value[0] <= config.lambda_low:
            raise ValueError('Lambda value is too low; examine the range for WFST Distance scaling.')

    
    optimal_beta_value = [hyperparameter_utils.get_optimal_hyperparameter_value_with_dict(this_split, this_dataset_name, beta_model_dict, model_class, 'beta')]
    if config.fail_on_beta_edge:
        if optimal_beta_value[0] >= config.beta_high:
            raise ValueError('Beta value is too high; examine the range for Levenshtein Distance scaling.')
        if optimal_beta_value[0] <= config.beta_low:
            raise ValueError('Beta value is too low; examine the range for Levenshtein Distance scaling.')

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
         
        best_beta_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, beta_model_dict, all_tokens_phono, optimal_beta_value[0], examples_mode, 'levdist')
        beta_tags = beta_model_dict['kwargs']['use_speaker_labels']
        beta_context_width = beta_model_dict['kwargs']['context_width_in_utts']
        beta_score_folder = hyperparameter_utils.load_hyperparameter_folder(this_split, this_dataset_name, beta_tags, beta_context_width, model_class)
        best_beta_scores.to_pickle(join(beta_score_folder, f'levdist_run_models_across_time_{age_str}.pkl'))
    
        best_lambda_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, lambda_model_dict, all_tokens_phono, optimal_lambda_value[0], examples_mode, 'wfst')
        lambda_tags = lambda_model_dict['kwargs']['use_speaker_labels']
        lambda_context_width = lambda_model_dict['kwargs']['context_width_in_utts']
        lambda_score_folder = hyperparameter_utils.load_hyperparameter_folder(this_split, this_dataset_name, lambda_tags, lambda_context_width, model_class)
        best_lambda_scores.to_pickle(join(lambda_score_folder, f'wfst_run_models_across_time_{age_str}.pkl'))
        # this will be the same as 76

    return best_beta_scores, best_lambda_scores
    
if __name__ == '__main__':
    
    start_time = str(datetime.today())
    
    parser = parsers.split_parser()
    parser.add_argument('--examples_mode', nargs = '?', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_args = vars(raw_args)
    
    query_model_str = load_models.get_model_id(
        split_name = this_args['split'],
        dataset_name = this_args['dataset'],
        with_tags =  this_args['use_tags'],
        context_width = this_args['context_width'],
        model_type = this_args['model_type'],
    )    
    
    print(f'examples mode: {this_args["examples_mode"]}')
                                                                    
    all_phono = load_splits.load_phono()
    this_sample_dict = load_splits.load_sample_model_across_time_args()
     
    best_beta_scores, best_lambda_scores = call_single_across_time_model(this_sample_dict, all_phono, this_args['model_type'], this_args['split'], this_args['dataset'], this_args['use_tags'], this_args['context_width'], this_args['examples_mode'])
    
    print(f'Computations complete for: {query_model_str}')
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
    
    
    