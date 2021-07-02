
import os
from os.path import join, exists
from utils import transfomers_bert_completions, split_gen

import pandas as pd


def get_beta_search_values(low = 2.5, high = 3.5, num_values = 10, grid = False):
    
    if not grid:
        # Random hyperparam search
        beta_samples = np.random.uniform(low, high, num_values)
    else: # Grid search
        test_beta_vals = np.arange(low, high, (high - low) / num_values)
    
    return beta_samples
