
import os
from os.path import join, exists
from utils import transfomers_bert_completions, split_gen

import pandas as pd

def sample_successes(split_name, dataset_name, data_dir, n = 5000, regenerate = False):
    
    # Try to load the data from the cache first.
    
    this_data_folder = split_gen.get_split_folder(split_name, dataset_name, data_dir)
    success_utts = pd.read_csv(join(this_data_folder, 'success_utts.csv'))
    this_data_path = join(this_data_folder, f'success_utts_beta_sampled_{n}.csv')
    
    if regenerate or not exists(this_data_path):
        # Need to sample the successes again and save them.
        success_utts_for_beta_fitting = success_utts.sample(n, replace=False).utterance_id
        success_utts_for_beta_fitting.to_csv(this_data_path)
    else:
        success_utts_for_beta_fitting = pd.read_csv(this_data_path)
    
    return success_utts_for_beta_fitting

def get_beta_search_values(low = 2.5, high = 3.5, num_values = 10, grid = False):
    
    if not grid:
        # Random hyperparam search
        beta_samples = np.random.uniform(low, high, num_values)
    else: # Grid search
        test_beta_vals = np.arange(low, high, (high - low) / num_values)
    
    return beta_samples

def x():
    pass
