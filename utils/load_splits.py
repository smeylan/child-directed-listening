# Code for loading the training data that has been split.


import os
from os.path import join, exists
 
from utils import split_gen, load_csvs
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

def get_sample_path(data_type, task_name, split_name, dataset_name, age = None):
    
    n = get_n(task_name)
    
    assert ( (age is None) and (task_name == 'beta' or split_name == 'all') ) or ( (age is not None) and (task_name == 'models_across_time') )
    age_str = f'_{float(age)}' if age is not None else ''
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    this_data_folder = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    this_data_path = join(this_data_folder, f'{data_type}_utts_{task_name}_{n}{age_str}.csv')
    
    return this_data_path


def get_all_ages_sample_paths(split_name, dataset_name):
    
    """
    Gets all of the sample paths for a given split.
    """
    this_data_folder = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    template = join(this_data_folder, f'*_utts_models_across_time_{config.n_across_time}_*.csv')
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


def get_which_sample_paths(split_name, dataset_name, search_for):
    
    has_data = lambda this_str : search_for in this_str
    this_paths = get_all_ages_sample_paths(split_name, dataset_name)
    return sorted(list(filter(has_data, this_paths)))
    
    
def get_success_sample_paths(split_name, dataset_name):
    return get_which_sample_paths(split_name, dataset_name, 'success')

def get_yyy_sample_paths(split_name, dataset_name):
    return get_which_sample_paths(split_name, dataset_name, 'yyy')


def get_all_ages_in_samples(split_name, dataset_name):
    """
    Gets all of the ages available in the sample for a given split.
    """
    
    age_dict = get_all_ages_sample_paths(split_name, dataset_name)
    
    return sorted(list(age_dict.keys()))
   
    
def sample_successes_yyy(pool, task, split, dataset, utts_pool, age):
    """
    task_name = designates the cached value to use for optimizations.
        The cache should be different for beta optimization and run_models_across_time.
    """
    
    
    if age is not None: # Sample per age
        utts_pool = utts_pool[utts_pool.year == age]
        
    num_samples = utts_pool.shape[0]
    n = min(num_samples, get_n(task))
    this_data_path = get_sample_path(pool, task, split, dataset, age)
    
    # Need to sample the successes again and save them.
    print(f"Resampling for: {pool}, {task}, {split}, {dataset}, age: {age}")
    sample = utts_pool.sample(n, replace=False).utterance_id
    sample.to_csv(this_data_path)
    
    return sample

def load_sample_successes(task, split, dataset, age = None):
    this_path = get_sample_path('success', task, split, dataset, age)
    return load_csvs.load_csv_with_lists(this_path)

def load_sample_yyy(task, split, dataset, age = None):
    
    this_path = get_sample_path('yyy', task, split, dataset, age)
    return load_csvs.load_csv_with_lists(this_path)


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
    
    
    
def load_eval_data_all(split_name, dataset_name):
    
    """
    7/15/21: Split out the loading logic to accomodate child loading -- should be orthogonal in the 
    non-child code.
    
    Loading cached data relevant to the model scoring functions in yyy analysis.
    Note that for children, this loads the entire split, not the eval split.
    
    (Loads Providence data)
    """
    return load_eval_data(split_name, dataset_name, '')



def load_eval_data(split_name, dataset_name, modifier):
    """
    modifier is used to 
    """
    phono_filename = f'{modifier}pvd_utt_glosses_phono_cleaned_inflated.pkl'
    success_utts_filename = f'{modifier}success_utts.csv'
    yyy_utts_filename = f'{modifier}yyy_utts.csv'

    data_filenames = [phono_filename, success_utts_filename, yyy_utts_filename]
    this_folder_path = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    
    data_name = {
       f'{modifier}pvd_utt_glosses_phono_cleaned_inflated.pkl' : 'phono',
       f'{modifier}success_utts.csv' : 'success_utts',
       f'{modifier}yyy_utts.csv' : 'yyy_utts',
    }
    
    data_dict = {}
    
    for f in data_filenames:
        this_path = join(this_folder_path, f)
        data_dict[data_name[f]] = load_csvs.load_csv_with_lists(this_path) if f.endswith('.csv') else pd.read_pickle(this_path)
    
    return data_dict



def load_child_eval_data(name):
    
    return load_eval_data('child', name, 'eval_')
    
    
    
