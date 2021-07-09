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

def get_sample_path(data_type, task_name, split_name, dataset_name):
    
    n = get_n(task_name)
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    this_data_folder = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    success_utts = load_csvs.load_csv_with_lists(join(this_data_folder, f'{data_type}_utts.csv'))
    this_data_path = join(this_data_folder, f'{data_type}_utts_{task_name}_{n}.csv')
    
    return this_data_path
    
def sample_successes_yyy(pool, task, split, dataset, utts_pool):
    """
    task_name = designates the cached value to use for optimizations.
        The cache should be different for beta optimization and run_models_across_time.
    """
    
    num_samples = utts_pool.shape[0]
    n = min(num_samples, get_n(task))
    this_data_path = get_sample_path(pool, task, split, dataset)
    
    # Need to sample the successes again and save them.
    print(f"Resampling for: {pool}, {task}, {split}, {dataset}")
    sample = utts_pool.sample(n, replace=False).utterance_id
    sample.to_csv(this_data_path)
    
    return sample

def load_sample_successes(task, split, dataset):
    this_path = get_sample_path('success', task, split, dataset)
    return load_csvs.load_csv_with_lists(this_path)

def load_sample_yyy(task, split, dataset):
    
    this_path = get_sample_path('yyy', task, split, dataset)
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
    Loading cached data relevant to the model scoring functions in yyy analysis.
    """
    
    phono_filename = 'pvd_utt_glosses_phono_cleaned_inflated.pkl'
    success_utts_filename = 'success_utts.csv'
    yyy_utts_filename = 'yyy_utts.csv'

    data_filenames = [phono_filename, success_utts_filename, yyy_utts_filename]
    this_folder_path = split_gen.get_split_folder(split_name, dataset_name, config.eval_dir)
    
    data_name = {
       'pvd_utt_glosses_phono_cleaned_inflated.pkl' : 'phono',
       'success_utts.csv' : 'success_utts',
       'yyy_utts.csv' : 'yyy_utts',
    }
    
    data_dict = {}
    
    for f in data_filenames:
        this_path = join(this_folder_path, f)
        data_dict[data_name[f]] = load_csvs.load_csv_with_lists(this_path) if f.endswith('.csv') else pd.read_pickle(this_path)
    
    return data_dict
    
    
    
    
