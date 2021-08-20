
from utils import split_gen, sampling, data_cleaning
import config

from collections import defaultdict

import math

# 7/23/21: https://www.mikulskibartosz.name/how-to-set-the-global-random_state-in-scikit-learn/
# Information reference, not for code -- sufficient to seed numpy for sklearn.

import numpy as np 

SEED = config.SEED
np.random.seed(SEED)

import sklearn
from sklearn import model_selection

def get_beta_idxs(pool, split_attr):
    
    sample = sampling.sample_pool_ids(pool, config.n_beta)
    
    return sample

def find_splits_across_ages(raw_pool):
    
    phase_idxs = defaultdict(list)
    
    for age in data_cleaning.get_years(raw_pool):
        
        # Sample one transcript per year, per phase
        
        pool = raw_pool[raw_pool.year == age]
        
        all_ids = list(set(pool.transcript_id))
        num_transcripts = len(all_ids)
        
        num_val_eval_req = config.child_val_eval_num * 2
        sample_n = min(num_transcripts, num_val_eval_req)
    
        # Prioritize test, then val, then train.
        
        # Train split
        if num_transcripts > num_val_eval_req:
            train_idxs, val_eval_idxs = split_gen.determine_split_idxs(pool, 'transcript_id', val_num = sample_n)
        else:
            train_idxs = np.array([])
            val_eval_idxs = np.array(all_ids)
            
        # Val/Eval split
        # Prioritize the eval split.
        
        num_val_eval_avail = val_eval_idxs.shape[0]
        
        if num_val_eval_avail == 1:
            val_idxs = np.array([])
            eval_idxs = val_eval_idxs
        else:
            ideal_eval_req = math.ceil(sample_n / 2)
            num_eval_real = min(num_val_eval_avail, ideal_eval_req)
            val_idxs, eval_idxs = sklearn.model_selection.train_test_split(val_eval_idxs, test_size = num_eval_real)
            
        for phase, phase_idx_set in zip(['train', 'val', 'eval'], [train_idxs, val_idxs, eval_idxs]):
            phase_idxs[phase].append(phase_idx_set)
            
    phase_idxs = { phase :  np.concatenate(phase_idxs[phase]) for phase in phase_idxs }
    
    return phase_idxs



def augment_with_all_subsamples(df, phase):

    for ideal_n in config.subsamples:
        
        # To allow it to pass the checks
        # + confirm all sample-able utterances receive either a yes or no sampling
    
        df.loc[df['phase_child_sample'].isna(), get_subsample_key(ideal_n)] = False
        df = augment_with_subsamples(df, phase, ideal_n)
    
    return df
        

def augment_with_subsamples(df, phase, ideal_n):
    """
    Intended for use with utterance lists that aren't already randomly sampled before saving,
        i.e. child scores for cross scoring.
    """
    
    utt_pool = np.unique(df[df.phase_child_sample == phase].utterance_id)
    
    n_avail = utt_pool.shape[0]
    n = min(n_avail, ideal_n)
    
    to_subsample = set(np.random.choice(utt_pool, size = (n,), replace = False))
    
    this_attr = get_subsample_key(ideal_n)
    
    df.loc[((df.utterance_id.isin(to_subsample)) & (df.phase_child_sample == phase)), this_attr] = True
    df.loc[((~df.utterance_id.isin(to_subsample)) & (df.phase_child_sample == phase)), this_attr] = False
    
    return df

def get_subsample_key(this_n):
    return f'phase_child_sample_n={this_n}'
    

def split_child_subsampling(all_phono):
    
    for phase in ['train', 'val', 'eval']:
        all_phono = augment_with_all_subsamples(all_phono, phase)
    
    return all_phono

        
