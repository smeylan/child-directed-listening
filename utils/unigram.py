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
    """
    
    split = 'all'; dataset = 'all'
    this_sample_successes = load_splits.load_sample_successes(task, split, dataset)
    this_sample_yyy = load_splits.load_sample_yyy(task, split, dataset)
    
    # You need to index into the utterances themselves in chi_phono -- get this from the sample functions.
    # What exactly is bert_token_ids and how to correctly create it?
    # Maybe 
    
    return pd.concat([this_sample_successes, this_sample_yyy]).bert_token_id

    
    
    