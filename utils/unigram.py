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
    Retrieves the equivalent of score_store[-1].bert_tokens_id
        used in the "set" check for limiting unigram distributions.
    Assumes that the order of the bert token ids doesn't matter (read the code to check that this value is used as a set)
    """
    
    tokens = load_splits.load_phono()
    
    this_sample_successes = load_splits.load_sample_successes(task, split, dataset)
    
    if task == 'models_across_time':
        this_sample_yyy = load_splits.load_sample_yyy(task, split, dataset)
        select_sample_id = pd.concat([this_sample_successes, this_sample_yyy])
    else:
        select_sample_id = this_sample_successes
        
    select_phono = tokens.loc[tokens.utterance_id.isin(select_sample_id.utterance_id)]
    success_mask_bert_ids = select_phono[select_phono['partition'] == 'success'].bert_token_id
    
    if task == 'models_across_time':
        failure_mask_bert_ids = select_phono.loc[select_phono.partition == 'yyy','bert_token_id']
        all_bert_ids = pd.concat([failure_mask_bert_ids, success_mask_bert_ids])
    else:
        all_bert_ids = success_mask_bert_ids
    
    return all_bert_ids
    
    

    
    
    