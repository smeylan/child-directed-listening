# Code for loading the training data that has been split.


import os
from os.path import join, exists
 
from utils import split_gen
import glob

import pandas as pd
import pickle

import config

import numpy as np

def get_ages_sample_paths(which_type):
    
    """
    Gets all of the sample paths for a given split.
    """

    data_folder = join(config.prov_dir, 'across_time_samples')
    template = join(data_folder, f'{which_type}_utts_models_across_time_{config.n_across_time}_*.csv')
    
    all_age_sample_paths = glob.glob(template)
    
    age2path = {}
    for path in all_age_sample_paths:
        # The age is located at the end.
        # 7/15/21: https://www.geeksforgeeks.org/python-os-path-splitext-method/
        filename = os.path.splitext(path)
        age = float(filename[0].split('_')[-2])
        # end cite
        age2path[age] = path
    
    return age2path
    
    
def get_age_success_sample_paths():
    return sorted(list(get_ages_sample_paths('success').values()))

def get_age_yyy_sample_paths():
    return sorted(list(get_ages_sample_paths('yyy').values()))

    
def load_sample_successes(task, split, dataset, age = None, eval_phase = config.eval_phase):
    this_path = get_sample_path('success', task, split, dataset, eval_phase, age)
    return pd.read_csv(this_path)


def load_sample_yyy(task, split, dataset, age = None, eval_phase = config.eval_phase):
    
    this_path = get_sample_path('yyy', task, split, dataset, eval_phase, age)
    return pd.read_csv(this_path)


def load_phono():
    
    return pd.read_pickle(join(config.prov_dir, 'pvd_all_tokens_phono_for_eval.pkl'))


    