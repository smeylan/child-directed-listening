# Functions for handling unigrams, in particular initializing them based on train distributions.

import pandas as pd
import numpy as np

import os
from os.path import join, exists

from utils import load_splits, split_gen

import configuration
config = configuration.Config()
import run_models_across_time
    

def get_sample_bert_token_ids(split = 'all', dataset = 'all'):
    """
    This is only intended for use with all/all split.
    This will retrieve all scoreable tokens inside of Providence.
    """
    
    all_phono = load_splits.load_phono()
    
    success_mask_bert_ids = all_phono[all_phono['partition'] == 'success'].bert_token_id
    failure_mask_bert_ids = all_phono.loc[all_phono.partition == 'yyy','bert_token_id']
    all_bert_ids = pd.concat([failure_mask_bert_ids, success_mask_bert_ids])
    
    return all_bert_ids
    
    

    
    
    