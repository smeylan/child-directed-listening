# 6/21/21 All code in this file refactored from Dr. Meylan's code
# Note to self: is the "true ids of the entries" check relevant here?

import os
from os.path import join, exists

import pandas as pd
import numpy as np

def get_split_folder(split_type, dataset_name, base_dir):
    
    path = join('data', join(base_dir, join(split_type, dataset_name)))
    
    if not exists(path):
        os.makedirs(path)
    
    return path

def fix_gloss(gloss):
    return(str(gloss).replace('+','').replace('_',''))


def get_token_frequencies(raw_data, split_type, dataset_name, data_base_path):
    
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

def get_chi_frequencies(raw_data, split_type, dataset_name, data_base_path):
    
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
    
def save_vocab(cleaned_glosses, split, dataset, base_dir):
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
    
    
    
def drop_errors(utt_data):
    
    utt_data['contains_error'] = ['xxx' in str(x) or 'yyy' in str(x) for x in utt_data.gloss]
    utt_data = utt_data.loc[~utt_data.contains_error]
    
    return utt_data.copy() # Avoid setting based on a slice of a dataframe.

def apply_gloss(utt_data):
    
    pass

def clean_glosses(data, fill_punct_val, verbose = False):
        
    punct_for_type = {
    'question':'?',
    'declarative':'.',
    'interruption':'!',
    'trail off':'...',
    'trail off question':'?',
    'imperative_emphatic':'!' 
    }
    
    assert fill_punct_val in ['.', None], "Tried to use a fill punctuation value that wasn't present in either yyy or finetune notebooks. For yyy behavior, use '.'. For finetune behavior, use None."
    
    data.gloss = [fix_gloss(x) for x in data.gloss]
    
    if verbose: print(data['type'].value_counts())
    
    # Cell 237
    data['punct'] = [punct_for_type[x] if x in punct_for_type else fill_punct_val
                        for x in data.type ]
    
    if verbose: print('Cell 238', data.iloc[0])
        
    # Cell 267
    data['speaker_code_simple'] = ['[CHI]' if x == 'CHI' else '[CGV]'
                                          for x in data.speaker_code]
    
    # Cell 268
    data = data.loc[[x is not None for x in data.punct]]
    data['gloss_with_punct'] = [x['speaker_code_simple'] + ' '+ x['gloss'].lower() + x['punct'] for x in data.to_dict('records')]
    
    return data
   
    
def prep_utt_glosses_for_split(data, fill_punct_val, verbose = False):
    
    """
    Highest level call.
 
    Cleans a given utt_glosses dataframe (to_clean_data) from its original query return
        (or close to original for children -- use the filtered chi_phono csv)
    Drops all types of errors.
    
    This function expects the raw query/utt_glosses dataframe equivalent.
    
    verbose controls whether or not printouts are active
        (for checking correctness relative to unfactored code on all/all split)
    """

    # Changed to drop xxx and yyy for all of the splits.
    
    if verbose: print('Cell 232 output', data.shape)
    
    # Cell 233 in the notebook relative to Dr Meylan's commit
    data = drop_errors(data)
    
    if verbose: print('Cell 233 output', data.shape)
        
    data = clean_glosses(data, None)
   
    if verbose: print('Cell 269', data.head(5).gloss_with_punct)
    
    # Cell 271: This was moved outside of token cleaning because it's needed for the CHI analysis.
    data['tokens'] = [str(x).lower().split(' ') for x in data.gloss]
    
    return data # Ready for input into token cleaning, get_chi_frequencies, get_token_frequencies
    

    

def split_glosses_shuffle(unsorted_cleaned_data, split_type, dataset_type, base_dir, val_ratio = None, val_num = None):
    """
    Highest level call.
    
    Randomly split the train and validation data by the number of unique transcript ids.
    Expects the output of prep_utt_glosses_for_split
    
    Note that split_type and dataset_type refer not to train/val, but:
        split_type = is this young or old? or is it part of the all data dataset?
        dataset_type = is this part of the age-based splits? or does it have all data?
        (child uses its own phase splitting logic)
    """
    
    assert (val_ratio is not None) ^ (val_num is not None), 'Exactly one of val_ratio and val_num should be specified.'
    
    data = unsorted_cleaned_data.copy()
    
    data = data.sort_values(by=['transcript_id', 'utterance_order'])
    
    # select 20 % of the transcripts for training
    
    transcript_inventory = np.unique(data.transcript_id)
    sample_num = val_num if val_ratio is None else int(val_ratio * len(transcript_inventory))
    validation_indices = np.random.choice(transcript_inventory, 
                                          sample_num)
    
    this_split_folder = get_split_folder(split_type, dataset_type, base_dir)
    
    data['partition'] = 'train'
    data.loc[data.transcript_id.isin(validation_indices),
             'partition'] = 'validation'
    
    data.loc[data.partition =='validation'] \
          [['gloss_with_punct']].to_csv(join(this_split_folder, 'validation.txt'), index=False, header=False)
    
    data.loc[data.partition =='train'] \
          [['gloss_with_punct']].to_csv(join(this_split_folder, 'train.txt'), index=False, header=False)
     
    print(f'Files written to {this_split_folder}')
    
    return data
       

def exec_split_gen(raw_data, split_name, dataset_name, base_dir, verbose = False):
    
    if not exists(base_dir):
        os.makedirs(base_dir)
    
    print('Beginning split gen call:', split_name, dataset_name)
    
    cleaned_utt_glosses = prep_utt_glosses_for_split(utt_glosses, verbose = verbose)
    
    tok_freq, chi_tok_freq = save_vocab(cleaned_utt_glosses, split_name, dataset_name, base_dir)

    if split_name == 'child':
        split_glosses_df = split_glosses_shuffle(cleaned_utt_glosses, split_name, dataset_name, base_dir, val_num = 400)
    else:
        assert split_name in ['all', 'age'], "Unrecognized split type argument. Should be one of 'all', 'age', or 'child'."
        split_glosses_df = split_glosses_shuffle(cleaned_utt_glosses, split_name, dataset_name, base_dir, val_ratio = 0.8)
    
    return split_glosses_df, tok_freq, chi_tok_freq
    
    
