
import os
from os.path import join, exists

from utils import load_models, load_splits, data_cleaning, parsers
from utils_model_sampling import beta_utils, sample_models_across_time

import numpy as np

import pandas as pd
import config

def load_sample_model_across_time_args(split_name, dataset_name):
    
    print('For child datasets, need to estimate beta based on all data available -- change this in the code.')
    
    sample_successes_id = load_splits.load_sample_successes('models_across_time', split_name, dataset_name)
    sample_yyy_id = load_splits.load_sample_yyy('models_across_time', split_name, dataset_name)
    
    eval_data_dict = load_splits.load_eval_data_all(split_name, dataset_name) 
    sel_sample_ids = pd.concat([sample_successes_id, sample_yyy_id])
    
    this_utts_with_ages_all = pd.concat([eval_data_dict['success_utts'], eval_data_dict['yyy_utts']])
    
    this_utts_with_ages_sample = this_utts_with_ages_all[this_utts_with_ages_all.utterance_id.isin(sel_sample_ids.utterance_id)]
    
    this_tokens_phono = eval_data_dict['phono']
        
    return this_utts_with_ages_sample, this_tokens_phono
    
    
def call_single_across_time_model(model_class, this_split, this_dataset_name, is_tags, context_width):
       
    assert model_class in {'childes', 'adult', 'flat_unigram', 'data_unigram'}, "Invalid model type presented."
   
    model_name = load_models.get_model_id(this_split, this_dataset_name, is_tags, context_width, model_class)
    
    all_models = load_models.get_model_dict()
    this_model_dict = all_models[model_name]
    
    utts, tokens = load_sample_model_across_time_args(this_split, this_dataset_name)
         
    # Load the optimal beta
    optimal_beta = beta_utils.get_optimal_beta_value(this_split, this_dataset_name, this_model_dict, model_class)
    
    utts = data_cleaning.augment_target_child_year(utts) # It expects things from the Providence notebook?
    
    ages = np.unique(utts.year)
   
    for idx, age in enumerate(ages):
        #for age in ages[:1]: # -> changing this to have more than one age will break the system -- why?
        
        percentage_done = idx / float(ages.shape[0]) * 100
        if int(percentage_done) % 5 == 0: print(f'{percentage_done}%') 
            
        this_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, utts, this_model_dict, tokens, beta_value = optimal_beta)
        
        this_tags = this_model_dict['kwargs']['use_speaker_labels']
        this_context_width = this_model_dict['kwargs']['context_width_in_utts']
        
        score_folder = beta_utils.load_beta_folder(this_split, this_dataset_name, this_tags, this_context_width, model_class)
        this_scores.to_csv(join(score_folder, f'run_models_across_time_{age}.csv'))# Need to assemble via model, then age later.
    
    return this_scores
    
if __name__ == '__main__':
    
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
        model_type = this_model_args['model_type']
    )
   
    # (model_class, this_split, this_dataset_name, is_tags, context_width):
    this_model_dict = load_models.get_model_dict()[query_model_str] # This is probably going to be slow, optimize later
    
    scores = call_single_across_time_model(this_model_args['model_type'], this_model_args['split'], this_model_args['dataset'], this_model_args['use_tags'], this_model_args['context_width'])
    
    print(f'Computations complete for: {query_model_str}')
    
    
    