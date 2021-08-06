# Functions for handling unigrams, in particular initializing them based on train distributions.

import pandas as pd
import numpy as np

import os
from os.path import join, exists

from utils import load_splits, split_gen

import config
    

def get_sample_bert_token_ids(task, split = 'all', dataset = 'all'):
    """
    This is only intended for use with all/all split.
    Retrieves the equivalent of score_store[-1].bert_token_id
        used in the "set" check for limiting unigram distributions.
    Assumes that the order of the bert token ids doesn't matter (read the code to check that this value is used as a set)
        
    You should check this function for correctness at the end.
    
    Note: 7/22/21 false alarm, these were limited to mask positions.
    """
    
    # The bert_token_ids are the bert_token_ids of every [MASK] in the sample.
    
    tokens = load_splits.load_phono()
    
    all_success_paths = load_splits.get_age_success_sample_paths()
    all_yyy_paths = load_splits.get_age_yyy_sample_paths()
    
    this_sample_successes = pd.concat([pd.read_csv(path)[['utterance_id']] for path in all_success_paths])
    this_sample_yyy = pd.concat([pd.read_csv(path)[['utterance_id']] for path in all_yyy_paths])

    select_sample_id = pd.concat([this_sample_successes, this_sample_yyy])
    select_phono = tokens.loc[tokens.utterance_id.isin(select_sample_id.utterance_id)]
    
    failure_mask_bert_ids = select_phono.loc[select_phono.partition == 'yyy','bert_token_id']
    success_mask_bert_ids = select_phono[select_phono['partition'] == 'success'].bert_token_id
    
    # Current general logic is correct per meeting
    all_bert_ids = pd.concat([failure_mask_bert_ids, success_mask_bert_ids])
    
    return all_bert_ids
    
    
    