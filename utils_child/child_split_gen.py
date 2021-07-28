
from utils import split_gen, sampling, data_cleaning
import config

from collections import defaultdict

import numpy as np
np.random.seed(config.SEED)

def get_beta_idxs(pool, split_attr, phase):
    
    sample = sampling.sample_pool_ids(pool, config.n_beta)
    
    return sample

def find_splits_across_ages(raw_pool):
    
    phase_idxs = defaultdict(list)
    
    for age in data_cleaning.get_years(raw_pool):
        
        # Sample one transcript per year, per phase
        
        pool = raw_pool[raw_pool.year == age]
        
        all_ids = list(set(pool.transcript_id))
        num_transcripts = len(all_ids)
        
        sample_n = min(num_transcripts, 2)
    
        # Prioritize test, then val, then train.
        
        # Train split
        if num_transcripts > 2:
            train_idxs, val_eval_idxs = split_gen.determine_split_idxs(pool, 'transcript_id', val_num = sample_n)
        else:
            train_idxs = np.array([])
            val_eval_idxs = np.array(all_ids)
            
        # Val/Eval split
        if val_eval_idxs.shape[0] == 1:
            eval_idxs = val_eval_idxs # Prioritize test split.
        else:
            val_choice = np.random.choice([0,1], 1).item()
            eval_choice = 1 if val_choice == 0 else 0
            
            eval_idxs = np.array([val_eval_idxs[eval_choice]])
            val_idxs = np.array([val_eval_idxs[val_choice]])
            
            print('eval', eval_idxs.shape)
            print('val', val_idxs.shape)
            phase_idxs['val'].append(val_idxs)
            
        phase_idxs['eval'].append(eval_idxs)
        phase_idxs['train'].append(train_idxs)
    
    phase_idxs = { phase :  np.concatenate(phase_idxs[phase]) for phase in phase_idxs }
    
    return phase_idxs

        
