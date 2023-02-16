# Code for loading the training data that has been split.
import os
from os.path import join, exists
import copy
import glob
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
from src.utils import split_gen, sampling, configuration, paths
config = configuration.Config()

    
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



def load_sample_model_across_time_args2(this_model_args):

    this_sample_dict = {}

    for age in np.arange(.5, 4.5, .5):

        success_utts_sample_path = paths.get_sample_csv_path(task_phase_to_sample_for='eval', split=this_model_args['test_split'], dataset=this_model_args['test_dataset'], data_type='success', age = age, n=config.n_beta)

        yyy_utts_sample_path = paths.get_sample_csv_path(task_phase_to_sample_for='eval', split=this_model_args['test_split'], dataset=this_model_args['test_dataset'], data_type='yyy', age = age, n=config.n_beta)
    
        success_utts = pd.read_csv(success_utts_sample_path)
        yyy_utts = pd.read_csv(yyy_utts_sample_path)

        this_age_dict = {'success': apply_if_subsample(success_utts),
            'yyy': apply_if_subsample(yyy_utts)}

        this_sample_dict[str(age)] = copy.copy(this_age_dict)

    return(this_sample_dict)


def load_phono():
    
    return pd.read_pickle(join(config.prov_dir, 'pvd_all_tokens_phono_for_eval.pkl'))


def get_child_names():
    """
    Get all Providence children.
    """
    
    all_phono = load_phono()
    return sorted(list(set(all_phono.target_child_name)))

