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
    

def get_sample_bert_token_ids(task):
    """
    This is really only intended for use with all/all split.
    Retrieves the equivalent of score_store[-1].bert_token_id
        used in the "set" check for limiting unigram distributions.
    Assumes that the order of the bert token ids doesn't matter (read the code to check that this value is used as a set)
        but attempts to keep the order in its design either way.
        
    You should check this function for correctness at the end.
    """
    
    split = 'all'; dataset = 'all'
    
    eval_data = load_splits.load_eval_data_all('all', 'all')
    tokens = eval_data['phono']
    
    this_sample_successes = load_splits.load_sample_successes(task, split, dataset)
    this_sample_yyy = load_splits.load_sample_yyy(task, split, dataset)
    
    select_sample_id = pd.concat([this_sample_successes, this_sample_yyy]).utterance_id
    
    select_phono = tokens.loc[tokens.id == select_sample_id]
    # This (select_phono) is equivalent to utt_df in the relevant code.
    
    failure_mask_positions = select_phono.loc[select_phono.token == 'yyy','token'].bert_token_ids
    
    # What does the double indexing do?
    success_mask_positions = (select_phono['partition'] == 'success').bert_token_ids
    
    # Avoid the original argwhere code because the indices will shift with the indexing.
    
    # Below line is good for conceptual guidance, but the actual "MASK" setting doesn't happen except inside 
    # the get stats for success code. So need to construct otherwise.
        
    # You need to index into the utterances themselves in chi_phono -- get this from the sample functions.
    # What exactly is bert_token_ids and how to correctly create it?
    # It's the following:
    # 'bert_token_id' : utt_df.loc[utt_df.token == '[MASK]'].bert_token_id}))
    # where utt-df is 
    # and all_tokens is phono. 
    
    return 

    
    
    