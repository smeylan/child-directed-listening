# Code for loading the training data that has been split.


import os
from os.path import join, exists
 
from utils import split_gen
import glob

import pandas as pd
import pickle

import config
    
    
def get_utts_from_ids(utts, utt_ids):
    
    return utts.loc[utts.utterance_id.isin(utt_ids)]

    
def get_n(task):
   
    assert task in ['beta', 'models_across_time'], "Invalid task name for sample successes -- use either 'beta' or 'models_across_time'."
    n = config.n_beta if task == 'beta' else config.n_across_time
    return n


def get_sample_path(data_type, task_name, split_name, dataset_name, eval_phase = config.eval_phase, age = None):
    
    n = get_n(task_name)
    
    print(age, task_name, split_name)
    
    assert ( (age is None) and (task_name == 'beta' or split_name == 'all') ) or ( (age is not None) and (task_name == 'models_across_time') )
    age_str = f'_{float(age)}' if age is not None else ''
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    if task_name == 'beta':
        this_data_folder = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    else:
        this_data_folder = join(config.eval_dir, 'across_time_samples')
        if not exists(this_data_folder):
            os.makedirs(this_data_folder)
            
    this_data_path = join(this_data_folder, f'{data_type}_utts_{task_name}_{n}{age_str}.csv')
    
    return this_data_path


def get_ages_sample_paths(which_type):
    
    """
    Gets all of the sample paths for a given split.
    """

    data_folder = join(config.eval_dir, 'across_time_samples')
    template = join(data_folder, f'{which_type}_utts_models_across_time_{config.n_across_time}_*.csv')
    
    all_age_sample_paths = glob.glob(template)
    
    age2path = {}
    for path in all_age_sample_paths:
        # The age is located at the end.
        # 7/15/21: https://www.geeksforgeeks.org/python-os-path-splitext-method/
        filename = os.path.splitext(path)
        age = float(filename[0].split('_')[-1])
        # end cite
        age2path[age] = path
    
    return age2path
    
    
def get_age_success_sample_paths():
    return sorted(list(get_ages_sample_paths('success').values()))

def get_age_yyy_sample_paths():
    return sorted(list(get_ages_sample_paths('yyy').values()))

    
    
def sample_pool_ids(this_pool, this_n):
    
    this_pool = np.unique(this_pool.id) # Enforce unique utterances.
    num_samples = this_pool.shape[0]
    
    n = min(num_samples, this_n)
    
    sample_ids = np.random.choice(this_pool, size = this_n, replace=False)
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

    print(f"Resampling for: {task}, {split}, {dataset}, age: {age}, phase: {eval_phase}")
    sample.to_csv(this_data_path) 
    
    return sample


def sample_successes(task, split, dataset, age, phono, eval_phase):
    
    success_pool = phono[phono.success_token]
    
    # Need to load the utts pool generally into the system -- how?
    sample = sample_successes_yyy(success_pool, 'success', age, task, split, dataset, eval_phase)
    
    return sample
    
    
def sample_yyy(task, split, dataset, age, phono, eval_phase):
    
    yyy_pool = phono[phono.yyy_token]
    
    # Need to load the utts pool generally into the system -- how?
    sample = sample_successes_yyy(yyy_pool, 'yyy', age, task, split, dataset, eval_phase)
    
    return sample
    
    
def load_sample_successes(task, split, dataset, age = None, eval_phase = config.eval_phase):
    this_path = get_sample_path('success', task, split, dataset, eval_phase, age)
    return pd.read_csv(this_path)

def load_sample_yyy(task, split, dataset, age = None, eval_phase = config.eval_phase):
    
    this_path = get_sample_path('yyy', task, split, dataset, eval_phase, age)
    return pd.read_csv(this_path)




def load_phono_successes_yyy_ids():
    
    return set(load_phono_successes_yyy().id)

def load_phono_successes_yyy():
    
    all_phono = load_phono()
    return all_phono[all_phono.success_token | all_phono.yyy_token]

def load_phono():
    
    return pd.read_pickle(join(config.eval_dir, 'pvd_all_tokens_phono_for_eval.pkl'))

##################
## TEXT LOADING ##
##################


def load_splits_folder_text(split):
    
    folders = glob.glob(join(config.data_dir, split) +'/*') # List the child names
    
    data = {}
    for path in folders:
        name = path.split('/')[-1]
        data[name] = load_split_text_path(split, name)
        
    return data


def load_split_text_path(split, dataset):
    
    # What else is needed?
    
    names = ['train', 'val', 'train_no_tags', 'val_no_tags']
    
    return {name : join(split_gen.get_split_folder(split, dataset, config.data_dir), f'{name}.txt')
           for name in names}



    