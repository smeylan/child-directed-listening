# Functions for handling unigrams, in particular initializing them based on train distributions.

import pandas as pd
import numpy as np

import os
from os.path import join, exists

from utils import load_splits, split_gen, load_csvs

import config
  

def load_chi_token_freq_all_split():
    
    # For now, only load unigrams from the all split.
    
    this_folder = split_gen.get_split_folder('all', 'all', config.data_dir)
    this_chi_vocab = load_csvs.load_csv_with_lists(join(this_folder, 'chi_vocab_train.csv'))
    
    return this_chi_vocab
    

def get_sample_bert_token_ids(task, split = 'all', dataset = 'all'):
    """
    This is only intended for use with all/all split.
    Retrieves the equivalent of score_store[-1].bert_token_id
        used in the "set" check for limiting unigram distributions.
    Assumes that the order of the bert token ids doesn't matter (read the code to check that this value is used as a set)
        but attempts to keep the order in its design either way.
        
    You should check this function for correctness at the end.
    """
    
    eval_data = load_splits.load_eval_data_all('all', 'all')
    tokens = eval_data['phono']

    # Need to extract all of the utterance ids in the entirety of the ages...
    # This will probably be very slow.
    
    all_success_paths = load_splits.get_success_sample_paths('all', 'all')
    all_yyy_paths = load_splits.get_yyy_sample_paths('all', 'all')
   
    
    # Use read_csv because you're just looking for utterance_ids
    this_sample_successes = pd.concat([pd.read_csv(path)[['utterance_id']] for path in all_success_paths])
    this_sample_yyy = pd.concat([pd.read_csv(path)[['utterance_id']] for path in all_yyy_paths])

    print('Fix this to be after filter')
    select_sample_id = pd.concat([this_sample_successes, this_sample_yyy])
    
    # Below are changes to the original code!
    # 7/7/21: Reference for isin usage
    # https://github.com/smeylan/child-directed-listening/blob/master/transfomers_bert_completions.py
    # Line 87
    select_phono = tokens.loc[tokens.id.isin(select_sample_id.utterance_id)]
    
    failure_mask_bert_ids = select_phono.loc[select_phono.token == 'yyy','bert_token_id']
    # I think this indexes the locations with yyy, then gets the attribute
    print('double check the use of loc and bert token id -- see comment above. in the code. probably on a small test case')

    success_mask_bert_ids = select_phono[select_phono['partition'] == 'success'].bert_token_id

    print('Correct logic generally -- double check the failure/success handling see the comment in the code.')
    # For a failure -- you are finding the yyy position.
    # For the successes, every single word in the utterance
    
    # Current general logic is correct per meeting
    all_bert_ids = pd.concat([failure_mask_bert_ids, success_mask_bert_ids])
    
    return all_bert_ids
    
    
    