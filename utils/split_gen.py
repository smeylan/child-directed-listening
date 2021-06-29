# 6/21/21 All code in this file refactored from Dr. Meylan's code
# Note to self: is the "true ids of the entries" check relevant here?

import os
from os.path import join, exists

import pandas as pd
import numpy as np

from utils import data_cleaning

SEED = 0
np.random.seed(SEED)


def get_age_split_data(data, months = 36):
    
    mask = data['target_child_age'] <= months * 30.5

    # Implied that target_child_age is in days,
    # and 30.5 days/month is used in the original Generalized Phonological analysis.

    young_df = data[mask]
    old_df = data[~mask]

    return young_df, old_df

def get_split_folder(split_type, dataset_name, base_dir = 'data/new_splits'):
   
    path = join(base_dir, join(split_type, dataset_name))
    
    if not exists(path):
        os.makedirs(path)
    
    return path


def get_token_frequencies(raw_data, split_type, dataset_name, data_base_path = 'data/new_splits'):
    
    """
    This is equivalent of Cell 271 - 277
    From Dr. Meylan: get the unigram counts for tokens and remap any problematic ones
    
    split_type = 'all, age, child': The type of data split, if any
    dataset_name = the name of the split. 'all', 'young', 'old', and the child names in lowercase
    
    Note: This expects cleaned "utt_glosses" (data) with all errors removed.
    """
    
    data = raw_data.copy()
    
    tokens = [y for x in data['tokens'] for y in x]
    
    token_frequencies = pd.Series(tokens).value_counts().reset_index()
    token_frequencies.columns = ['word','count']
    
    # Note: The vocab will be different for different splits of the data! Need to include a "dataset_name"
    
    this_folder = get_split_folder(split_type, dataset_name, data_base_path)
    token_frequencies.to_csv(join(this_folder, 'vocab.csv'))
    
    # Omit in_dict attribute: doesn't seem to be in any of the other notebooks/major py files
   
    return token_frequencies # Sort separately in the train/validation split if you want?

def get_chi_frequencies(raw_data, split_type, dataset_name, data_base_path = 'data/new_splits'):
    
    """
    Note: These are frequencies, so it needs to be different per sub-dataset
        (i.e. each piece of each split)
        general "chi_vocab.csv" can't be used for the sub-analyses
        
    Note: This expects cleaned "utt_glosses" (data) with all errors removed.
    """
    
    this_folder = get_split_folder(split_type, dataset_name, data_base_path)
    
    data = raw_data.copy()
    
    chi_data = data.loc[data.speaker_code == 'CHI']
    tokens = [y for x in chi_data['tokens'] for y in x]
    token_frequencies = pd.Series(tokens).value_counts().reset_index()
    token_frequencies.columns = ['word','count']
    token_frequencies.to_csv(join(this_folder, 'chi_vocab.csv'))
   
    return token_frequencies
    
def save_vocab(cleaned_glosses, split, dataset, base_dir = 'data/new_splits'):
    """
    
    Highest level call.
    
    Save the respective vocabulary files for this dataset split
    Expects the output of prep_utt_glosses_for_split.
        (i.e. cleaned and no speaking errors)
    
    Returns the freq. csv for tokens, child speaker tokens
    """
    
    tok_freq = get_token_frequencies(cleaned_glosses, split, dataset, base_dir)
    chi_freq = get_chi_frequencies(cleaned_glosses, split, dataset, base_dir)
    
    return tok_freq, chi_freq
    

def glosses_random_split(unsorted_cleaned_data, val_ratio = None, val_num = None):
    """
    Split off a subset of a pool of data to be used as validation.
    """
    
    assert (val_ratio is not None) ^ (val_num is not None), 'Exactly one of val_ratio and val_num should be specified.'
    
    data = unsorted_cleaned_data.copy()
    
    data = data.sort_values(by=['transcript_id'])
    
    # select 20 % of the transcripts for training
    
    transcript_inventory = np.unique(data.transcript_id)
    sample_num = val_num if val_ratio is None else int(val_ratio * len(transcript_inventory))
    validation_idx = np.random.choice(transcript_inventory, 
                                          sample_num)
    
    return validation_idx


def write_data_partitions_text(data_pool, split_folder, validation_indices):
    
    data_pool['phase'] = 'train'
    data_pool.loc[data_pool.transcript_id.isin(validation_indices),
             'phase'] = 'validation'
    
    train_df = data_pool.loc[data_pool.phase =='train']
    val_df = data_pool.loc[data_pool.phase =='validation']
    
    val_df[['gloss_with_punct']].to_csv(join(split_folder, 'validation.txt'), index=False, header=False)
    
    train_df[['gloss_with_punct']].to_csv(join(split_folder, 'train.txt'), index=False, header=False)
     
    print(f'Files written to {this_split_folder}')
    
    return data_pool, train_df, val_df
    
    
def split_glosses_shuffle(unsorted_cleaned_data, split_type, dataset_type, base_dir = 'data/new_splits', val_ratio = None, val_num = None):
    """
    
    Randomly split the train and validation data by the number of unique transcript ids.
    Expects the output of prep_utt_glosses_for_split
    
    Note that split_type and dataset_type refer not to train/val, but:
        split_type = is this young or old? or is it part of the all data dataset?
        dataset_type = is this part of the age-based splits? or does it have all data?
        (child uses its own phase splitting logic)
    """
    
    this_split_folder = get_split_folder(split_type, dataset_type, base_dir)
    validation_indices = glosses_random_split(unsorted_cleaned_data, val_ratio = val_ratio, val_num = val_num)
    
    data = write_data_partitions_text(data, this_split_folder, validation_indices) 
    
    return data
       

# This will perform all of the cleaning, splits, etc.
# I would just do repetitive cleaning.

def exec_split_gen(raw_data, split_name, dataset_name, base_dir = 'data/new_splits', verbose = False):
    
    """
    This should be executed for all and age splits -- child splits are external.
    """
    
    if not exists(base_dir):
        os.makedirs(base_dir)
    
    print('Beginning split gen call:', split_name, dataset_name)
    
    # Note: yyy uses "." as the default punct val. Splits use "None" as the default punct val.
    cleaned_utt_glosses = data_cleaning.prep_utt_glosses(raw_data, None, verbose = verbose)
    
    tok_freq, chi_tok_freq = save_vocab(cleaned_utt_glosses, split_name, dataset_name, base_dir)

    assert split_name in ['all', 'age'], "Unrecognized split type argument. Should be one of 'all', 'age', or 'child'."
    split_glosses_df = split_glosses_shuffle(cleaned_utt_glosses, split_name, dataset_name, base_dir, val_ratio = 0.8)
    
    return split_glosses_df, tok_freq, chi_tok_freq
    
    
