# Functions for handling unigrams, in particular initializing them based on train distributions.

import pandas as pd
import numpy as np

import os
from os.path import join, exists

from utils import load_splits, split_gen

import configuration
config = configuration.Config()
import run_models_across_time
    

def get_sample_bert_token_ids(task, split = 'all', dataset = 'all'):
    """
    This is only intended for use with all/all split.
    Retrieves the equivalent of score_store[-1].bert_tokens_id
        used in the "set" check for limiting unigram distributions.
    Assumes that the order of the bert token ids doesn't matter (read the code to check that this value is used as a set)
    """
    
    assert task in {'beta', 'models_across_time'}
    
    tokens = load_splits.load_phono()
    
    if task == 'beta':
        this_sample_successes = load_splits.load_sample_successes(split, dataset)
        select_sample_id = this_sample_successes
    
    else:
        all_ages_samples = load_splits.load_sample_model_across_time_args()
        
        all_samples = [] 
        
        for age in all_ages_samples:
            for this_type in ['success', 'yyy']:
                all_samples.append(all_ages_samples[age][this_type])
        select_sample_id = pd.concat(all_samples)
        
    select_phono = tokens.loc[tokens.utterance_id.isin(select_sample_id.utterance_id)]
    success_mask_bert_ids = select_phono[select_phono['partition'] == 'success'].bert_token_id
    
    if task == 'beta':
        all_bert_ids = success_mask_bert_ids
    else:
        failure_mask_bert_ids = select_phono.loc[select_phono.partition == 'yyy','bert_token_id']
        all_bert_ids = pd.concat([failure_mask_bert_ids, success_mask_bert_ids])
    
    return all_bert_ids
    
    

    
    
    