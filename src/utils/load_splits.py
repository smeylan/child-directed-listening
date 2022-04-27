# Code for loading the training data that has been split.
import os
from os.path import join, exists
import glob
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
from src.utils import split_gen, sampling, configuration
config = configuration.Config()

def get_ages_sample_paths(which_type, phase):
    
    """
    Gets all of the sample paths for a given split.
    """

    data_folder = join(config.prov_dir, 'across_time_samples')
    template = join(data_folder, f'{which_type}_utts_models_across_time_{config.n_across_time}_*_{phase}.csv')
    
    all_age_sample_paths = glob.glob(template)
    
    age2path = {}
    for path in all_age_sample_paths:
        # The age is located at the end.
        # 7/15/21: https://www.geeksforgeeks.org/python-os-path-splitext-method/
        filename = os.path.splitext(path)
        age = filename[0].split('_')[-2]
        # end cite
        age2path[age] = path
    
    return age2path

    
def apply_if_subsample(data, path = None):
    """
    Applies subsampling logic for either development purposes or using a smaller sample than n = 500.
    Because the utterances were originally randomly sampled, taking a prefix of a random sample should also be a random sample.
    """
    trunc_mode = (config.dev_mode or config.subsample_mode)
    
    assert config.n_beta == config.n_across_time, "Assumption for apply if subsample to hold."
    
    trunc_to_ideal = config.n_beta if not trunc_mode else config.n_subsample
    trunc_to =  min(trunc_to_ideal, data.shape[0])
    
    trunc_data = data.iloc[0:trunc_to]
            
    return trunc_data
    
    
def get_age_success_sample_paths(phase = config.eval_phase):
    raise ValueError('Deprecated')
    return sorted(list(get_ages_sample_paths('success', phase).values()))


def get_age_yyy_sample_paths(phase = config.eval_phase):
    raise ValueError('Deprecated')
    return sorted(list(get_ages_sample_paths('yyy', phase).values()))


def load_sample_successes(split, dataset, age = None, eval_phase = config.eval_phase):
    raise ValueError('Deprecated')
    this_path = sampling.get_sample_path('success', 'beta', split, dataset, eval_phase, age)
    this_data = pd.read_csv(this_path)
    return apply_if_subsample(this_data)


def load_sample_model_across_time_args():
    
    sample_dict = defaultdict(dict)
    
    success_paths = get_ages_sample_paths('success', config.eval_phase)
    yyy_paths = get_ages_sample_paths('yyy', config.eval_phase)
    
    for name, path_set in zip(['success', 'yyy'], [success_paths, yyy_paths]):
        for age, path in path_set.items():
            this_data = pd.read_csv(path)
            this_data = apply_if_subsample(this_data)

            sample_dict[age][name] = this_data
        
    return sample_dict

def load_phono():
    
    return pd.read_pickle(join(config.prov_dir, 'pvd_all_tokens_phono_for_eval.pkl'))


    