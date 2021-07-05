
# 6/21/21 All code in this file refactored from Dr. Meylan's code
# Note to self: is the "true ids of the entries" check relevant here?

import os
from os.path import join, exists

import pandas as pd
import numpy as np

import config

def filter_speaker_tags(this_df):
    """
    Used for loading appropriate data for the run models across time functions.
    """
    filtered_df = this_df[this_df.token != '[cgv]']
    filtered_df = filtered_df[this_df.token != '[chi]']
    
    assert all(' ' not in token for token in this_df.token), "There is a multiword token, check if it has speaker tags."
    
    return filtered_df

def gloss_df_augmentation(tokens_df, utt_ids):
    """
    Used for augmenting the utts_with_age dataframe with the joined gloss.
    Used for writing the train/val txt files for child splits.
    """
    glosses = []
    for i, utt_id in enumerate(utt_ids):
        if i % 10000 == 0: print(f'Computing gloss df augmentation, {round(i / len(utt_ids) * 100.0, 4)}% complete.')
        glosses.append(join_gloss_tokens(utt_id, tokens_df))
    return pd.DataFrame.from_dict({'utterance_id' : utt_ids, 'gloss' : glosses})
    
    
def join_gloss_tokens(utt_id, all_tokens_df):
    """
    Joins the tokens given per entry of the query of Providence data in their token order, such that they represent the tokens of utt_id.
    """
    
    entry = all_tokens_df[all_tokens_df.utterance_id == utt_id]
       
    # Assume that all_tokens_df will append each token in token_order, and check that this is true.
    composed_gloss = []; order_gloss = []
    for g, idx in zip(entry.gloss, entry.token_order):
        composed_gloss.append(g)
        order_gloss.append(idx)
       
    assert order_gloss == list(range(1, len(entry.gloss) + 1)), "Tokens were not joined in ascending order."
    
    return ' '.join(composed_gloss)
   
def fix_gloss(gloss):
    return(str(gloss).replace('+','').replace('_',''))
 
def drop_errors(utt_data):
    
    utt_data['contains_error'] = ['xxx' in str(x) or 'yyy' in str(x) for x in utt_data.gloss]
    utt_data = utt_data.loc[~utt_data.contains_error]
    
    return utt_data.copy() # Avoid setting based on a slice of a dataframe.



def clean_glosses(data, fill_punct_val):
        
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
    
    if config.verbose: print(data['type'].value_counts())
    
    # Cell 237
    data['punct'] = [punct_for_type[x] if x in punct_for_type else fill_punct_val
                        for x in data.type ]
    
    if config.verbose: print('Cell 238', data.iloc[0])
        
    # Cell 267
    data['speaker_code_simple'] = ['[CHI]' if x == 'CHI' else '[CGV]'
                                          for x in data.speaker_code]
    
    # Cell 268
    data = data.loc[[x is not None for x in data.punct]]
    data['gloss_with_punct'] = [x['speaker_code_simple'] + ' '+ x['gloss'].lower() + x['punct'] for x in data.to_dict('records')]
    
    return data


   
def prep_utt_glosses(data, fill_punct_val):
    
    """
    Highest level call for converting and augmenting raw queried data.
 
    Cleans a given utt_glosses dataframe (to_clean_data) from its original query return
        (or close to original for children -- use the filtered chi_phono csv)
    Drops all types of errors.
    
    This function expects the raw query/utt_glosses dataframe equivalent.
    
    verbose controls whether or not printouts are active
        (for checking correctness relative to unfactored code on all/all split)
    """

    # Changed to drop xxx and yyy for all of the splits.
    
    if config.verbose: print('Cell 232 output', data.shape)
    
    # Cell 233 in the notebook relative to Dr Meylan's commit
    data = drop_errors(data)
    
    if config.verbose: print('Cell 233 output', data.shape)
        
    data = clean_glosses(data, None)
   
    if config.verbose: print('Cell 269', data.head(5).gloss_with_punct)
    
    # Cell 271: This was moved outside of token cleaning because it's needed for the CHI analysis.
    data['tokens'] = [str(x).lower().split(' ') for x in data.gloss]
    
    return data # Ready for input into token cleaning, get_chi_frequencies, get_token_frequencies

