import os
from os.path import join, exists
import pandas as pd
import numpy as np
from src.utils import split_gen, configuration, paths
config = configuration.Config()
np.random.seed(config.SEED)

    
def get_n(task_phase_to_sample_for):

    assert task_phase_to_sample_for in ['fit', 'eval'], "Invalid task name for sample successes -- use either 'fit' or 'eval'."
    n = config.n_beta if task_phase_to_sample_for == 'fit' else config.n_across_time
    return n


def sample_pool_ids(this_pool, this_n):
    
    this_pool = np.unique(this_pool.utterance_id) # Enforce unique utterances.
    
    num_samples = this_pool.shape[0]
    
    n = min(num_samples, this_n)
    
    sample_ids = np.random.choice(this_pool, size = n, replace=False)
    sample = pd.DataFrame.from_records({'utterance_id' : sample_ids.tolist()})
    
    return sample
    
    
def sample_successes_yyy(pool, task_phase_to_sample_for, split, dataset, data_type, age,  n = None):
        
    if n is None:
        n = get_n(task_phase_to_sample_for)
    
    if age is not None: # Sample per age
        pool = pool[pool.year == age]
     
    # Need to sample the successes again and save them.
    # Use CSV for compatibility 
    
    sample = sample_pool_ids(pool, n)
    
    this_data_path = paths.get_sample_csv_path(task_phase_to_sample_for, split, dataset, data_type, age, n)    

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
    
def sample_successes(task_phase_to_sample_for, test_split, test_dataset, age, raw_phono):
    
    phono = _filter_for_scoreable_without_partition(raw_phono)
    success_pool = phono[phono.partition == 'success']
    
    sample = sample_successes_yyy(success_pool, task_phase_to_sample_for, test_split, test_dataset, 'success', age)
    
    return sample
    
    
def sample_yyy(task_phase_to_sample_for, test_split, test_dataset, age, raw_phono):
    
    phono = _filter_for_scoreable_without_partition(raw_phono)
    yyy_pool = phono[phono.partition == 'yyy']
    
    sample = sample_successes_yyy(yyy_pool, task_phase_to_sample_for, test_split, test_dataset, 'yyy', age)
    
    return sample
