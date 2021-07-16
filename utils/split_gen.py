# 6/21/21 All code in this file refactored from Dr. Meylan's code
# Note to self: is the "true ids of the entries" check relevant here?

import os
from os.path import join, exists

import pandas as pd
import numpy as np

import sklearn
from sklearn import model_selection

from utils import data_cleaning
import config

SEED = config.SEED
np.random.seed(SEED)


def get_age_split_data(raw_data, months = config.age_split):
    
    data = raw_data.dropna(subset = ['target_child_age'])
    
    young_mask = data['target_child_age'] <= months * 30.5
    old_mask = data['target_child_age'] > months * 30.5
    
    # Implied that target_child_age is in days,
    # and 30.5 days/month is used in the original Generalized Phonological analysis.

    young_df = data[young_mask]
    old_df = data[old_mask]

    return young_df, old_df

def get_split_folder(split_type, dataset_name, base_dir):
   
    path = join(base_dir, join(split_type, dataset_name))
    
    if not exists(path):
        os.makedirs(path)
    
    return path

# Removed 'vocab.csv' because it isn't used in latest run_mlm.py code.


def save_chi_vocab(train_data, split_type, dataset_name):
    
    """
    Note: These are frequencies, so it needs to be different per sub-dataset
        (i.e. each piece of each split)
        used to initialize the unigrams.
        general "chi_vocab.csv" can't be used for the sub-analyses
        
    Note: This expects cleaned "utt_glosses" (data) with all errors removed.
    
    Note: If you load train_data from a saved df, it will convert list -> str which will mess up the token identification.
    'token' should be re-cast to list for correct behavior, as seen below.
    """
    
    this_folder = get_split_folder(split_type, dataset_name, config.data_dir)
    
    chi_data = train_data.loc[train_data.speaker_code == 'CHI']
    
    #raw_tokens = list(chi_data['tokens'])
    
    #cast2list = lambda str_list : eval(str_list) # See the function comment for why
    #eval_tokens = list(map(cast2list, raw_tokens)) if isinstance(raw_tokens[0], str) else raw_tokens
    
    #tokens = [y for x in eval_tokens for y in x]
    
    tokens = [y for x in chi_data['tokens'] for y in x]
    
    token_frequencies = pd.Series(tokens).value_counts().reset_index()
    token_frequencies.columns = ['word','count']
    
    token_frequencies.to_csv(join(this_folder, 'chi_vocab_train.csv'))
    
    # Do not use things marked chi_vocab, they are from the past version.
   
    return token_frequencies

    

def determine_split_idxs(unsorted_cleaned_data, split_on, val_ratio = None, val_num = None):
    """
    Split off a subset of a pool of data to be used as validation.
    This function was changed from the original.
    
    If you want to use a model with context, transformers_bert_completions requires transcript id to be the splitting attr
        for disjointedness guarantees.
    If you want to use a child model,
        split on utterance id, and set context argument = None, to never use context (otherwise the eval diversity is too small).
    """
    
    assert (val_ratio is not None) ^ (val_num is not None), 'Exactly one of val_ratio and val_num should be specified.'
    
    data = unsorted_cleaned_data.copy()
    data = data.sort_values(by=[split_on])
    
    split_attr_inventory = np.unique(data[split_on])
    sample_num = val_num if val_ratio is None else int(val_ratio * len(split_attr_inventory))
    
    train_idx, validation_idx = sklearn.model_selection.train_test_split(split_attr_inventory, test_size = sample_num)
    
    return train_idx, validation_idx 
    

    
def find_phase_data(phase, pool):

    this_phase_data = pool.loc[pool.phase == phase]
    return this_phase_data


def assign_and_find_phase_data(phase, split_on, phase_idxs, data_pool):
    """
    Different from the original function, re-test
    See dtermine_split_idxs comments on what to split on for which models.
        basically child = utterance_id, anything else = transcript_id.
    """
    
    data_pool.loc[data_pool[split_on].isin(phase_idxs),
             'phase'] = phase
    
    phase_data = find_phase_data(phase, data_pool)
    
    return phase_data, data_pool

def find_in_phase_idxs(data_pool, phase_idxs, split_on):
    """
    Added, for use in child -- but didn't refactor age/all code.
    """
    return data_pool.loc[data_pool[split_on].isin(phase_idxs)]


def filter_text(text_path):
    
    remove_tags = lambda this_str : this_str.replace('[CHI] ', ''). replace('[CGV] ', '')
    with open(text_path, 'r') as f:
        all_str = list(map(remove_tags, f.readlines()))
        
    return all_str

    
def write_data_partitions_text(all_data, split_folder, phase, phase_idxs, split_on):
    """
    See determine_split_idxs comments on what to use for split_on argument per split type.
    Need to test this function, it has changed.
    """
    
    phase_data, all_data_with_assignments = assign_and_find_phase_data(phase, split_on, phase_idxs, all_data)
    
    this_file_path = join(split_folder, f'{phase}.txt')
    phase_data[['gloss_with_punct']].to_csv(this_file_path, index=False, header=False)
     
    print(f'File written to {this_file_path}')
    
    # Write the tagless version as well.
    filtered_text = filter_text(this_file_path)
    
    # Separate the filename and modify it.
    filtered_path = join('/'.join(this_file_path.split('/')[:-1]), f'{phase}_no_tags.txt')
    with open(filtered_path, 'w') as f:
        f.writelines(filtered_text)
        
    print(f'File written to {filtered_path}')
    
    return all_data_with_assignments, phase_data
    
    
def split_glosses_shuffle(unsorted_cleaned_data, split_type, dataset_type, split_on, base_dir = 'data/new_splits', val_ratio = None, val_num = None):
    """
    
    Randomly split the train and validation data by the number of unique transcript ids.
    Expects the output of prep_utt_glosses_for_split
    
    Note that split_type and dataset_type refer not to train/val, but:
        split_type = is this young or old? or is it part of the all data dataset?
        dataset_type = is this part of the age-based splits? or does it have all data?
        (child uses its own phase splitting logic)
    """
    
    this_split_folder = get_split_folder(split_type, dataset_type, base_dir)
    train_idxs, val_idxs = determine_split_idxs(unsorted_cleaned_data, split_on, val_ratio = val_ratio, val_num = val_num)
    
    data, train_df = write_data_partitions_text(data, this_split_folder, 'train', train_idxs, split_on)
    data, val_df = write_data_partitions_text(data, this_split_folder, 'val', val_idxs, split_on)
    
    return data, train_df, val_df
      

def exec_split_gen(raw_data, split_name, dataset_name):
    
    """
    This should be executed for all and age splits -- child splits are external.
    """
    
    assert split_name in ['all', 'age'], "Unrecognized split type argument. Should be one of 'all' or 'age'. Don't use child with this function."

    if not exists(config.data_dir):
        os.makedirs(config.data_dir)
    
    this_split_folder = get_split_folder(split_name, dataset_name, config.data_dir)
    
    print('Beginning split gen call:', split_name, dataset_name)
    
    # Note: yyy uses "." as the default punct val. Splits use "None" as the default punct val.
    cleaned_utt_glosses = data_cleaning.prep_utt_glosses(raw_data, None)
    
    train_idxs, val_idxs = determine_split_idxs(cleaned_utt_glosses, 'transcript_id', val_ratio = config.val_ratio)
    
    split_glosses_df, train_df = write_data_partitions_text(cleaned_utt_glosses, this_split_folder, 'train', train_idxs, 'transcript_id')
    split_glosses_df, val_df = write_data_partitions_text(cleaned_utt_glosses, this_split_folder, 'val', val_idxs, 'transcript_id')
    
    # chi_tok_freq = save_chi_vocab(train_df, split_name, dataset_name)
    # Need to re-integrate this line later when everything is run at once.
    # Along with the writing tagless files etc.
    
    return split_glosses_df, chi_tok_freq
    
    
