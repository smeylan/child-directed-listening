
import os
from os.path import join, exists

from utils import split_gen
import config

import numpy as np
np.random.seed(config.SEED)

import pandas as pd
    
def get_n(task):
   
    assert task in ['beta', 'models_across_time'], "Invalid task name for sample successes -- use either 'beta' or 'models_across_time'."
    n = config.n_beta if task == 'beta' else config.n_across_time
    return n


def get_sample_path(data_type, task_name, split_name, dataset_name, eval_phase = config.eval_phase, age = None):
    
    n = get_n(task_name)
    
    assert ( (age is None) and (task_name == 'beta' or split_name == 'all') ) or ( (age is not None) and (task_name == 'models_across_time') )
    age_str = f'_{float(age)}' if age is not None else ''
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    if task_name == 'beta':
        this_data_folder = split_gen.get_split_folder(split_name, dataset_name, config.prov_dir)
    else:
        this_data_folder = join(config.prov_dir, 'across_time_samples')
        if not exists(this_data_folder):
            os.makedirs(this_data_folder)
            
    this_data_path = join(this_data_folder, f'{data_type}_utts_{task_name}_{n}{age_str}_{eval_phase}.csv')
    
    return this_data_path


def sample_pool_ids(this_pool, this_n):
    
    this_pool = np.unique(this_pool.utterance_id) # Enforce unique utterances.
    
    num_samples = this_pool.shape[0]
    
    n = min(num_samples, this_n)
    
    sample_ids = np.random.choice(this_pool, size = n, replace=False)
    sample = pd.DataFrame.from_records({'utterance_id' : sample_ids.tolist()})
    
    return sample
    
    
def sample_successes_yyy(pool, data_type, age, task, split, dataset, eval_phase, n = None):
    """
    task_name = designates the cached value to use for optimizations.
        The cache should be different for beta optimization and run_models_across_time.
    """
    
    if n is None:
        n = get_n(task)
    
    if age is not None: # Sample per age
        pool = pool[pool.year == age]
     
    # Need to sample the successes again and save them.
    # Use CSV for compatibility 
    
    sample = sample_pool_ids(pool, n)
    
    this_data_path = get_sample_path(data_type, task, split, dataset, eval_phase, age)

    print(f"Resampling for: task: {task}, split: {split}, dataset: {dataset}, age: {age}, phase: {eval_phase}")
    sample.to_csv(this_data_path) 
    
    return sample



def _filter_for_scoreable_without_partition(df):
    """
    Filters for attributes that make a token scoreable, 
        except for partition requirement.
    """
    
    df = df[(df.actual_phonology != '')
           & (df.model_phonology != '')
           & (df.speaker_code == 'CHI')]

    return df
    
def sample_successes(task, split, dataset, age, raw_phono, eval_phase):
    
    phono = _filter_for_scoreable_without_partition(raw_phono)
    success_pool = phono[phono.partition == 'success']
    
    sample = sample_successes_yyy(success_pool, 'success', age, task, split, dataset, eval_phase)
    
    return sample
    
    
def sample_yyy(task, split, dataset, age, raw_phono, eval_phase):
    
    phono = _filter_for_scoreable_without_partition(raw_phono)
    yyy_pool = phono[phono.partition == 'yyy']
    
    sample = sample_successes_yyy(yyy_pool, 'yyy', age, task, split, dataset, eval_phase)
    
    return sample
    
