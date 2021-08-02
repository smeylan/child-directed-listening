
import os
from os.path import join, exists

import pandas as pd
import numpy as np

import config

import math

from utils import load_models, load_splits


def find_transcripts_with_successes_and_yyy(df):
    
    success_df_ids = set(df[df.partition == 'success'].transcript_id)
    yyy_df_ids = set(df[df.partition == 'yyy'].transcript_id)
    
    df = df[df.transcript_id.isin(success_df_ids & yyy_df_ids)]
    
    return df


def get_years(df):
    
    return sorted(list(set(df['year'].dropna())))

def cut_context_df(df, MAX_LEN = 512):
    
    # This can be really slow if you load the tokenizer every time
    tokenizer = load_models.get_primary_tokenizer()
    
    # 7/17/21: For advice on truncation, not code
    # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification

    # 7/18/21: For info on pandas behavior with indexing, not code
    # https://stackoverflow.com/questions/49962417/why-does-loc-have-inclusive-behavior-for-slices
    
    # 7/17/21: For info on BERT max length, not code
    #https://arxiv.org/pdf/1905.05583.pdf

    TARGET_LENGTH = MAX_LEN - 2
    # Because need space for CLS, SEP later
    # 512 is the max length for BERT.

    curr_len = df.shape[0]
    
    num_remove = curr_len - TARGET_LENGTH
    
    tokens_remove_front = num_remove // 2
    tokens_remove_back = math.ceil(num_remove / 2)
    
    start_idx = tokens_remove_front
    end_idx = curr_len - tokens_remove_back
    # Don't use negative indexing, it can cause weird bugs 
    
    sep_token_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    df = df.iloc[start_idx:end_idx]
    
    # Checked that token_id is int
    if df.iloc[-1]['token_id'] == sep_token_id: 
        df = df[:-1] # It seems that pandas has exclusive endpoint?
    if df.iloc[0]['token_id'] == sep_token_id:
        df = df[1:]
        # Exclude the first/final SEP in case it causes surprisal issues.

    return df


def augment_target_child_year(df):
    """
    Expects all_tokens_phono from Providence notebook.
    """
    df['year'] = .5*np.floor(df['target_child_age'] / (365. /2) ) 
    return df
    

def filter_speaker_tags(this_df):
    """
    Used for loading appropriate data for the run models across time functions.
    """
    filtered_df = this_df[this_df.token != '[cgv]']
    filtered_df = filtered_df[this_df.token != '[chi]']
    
    assert all(' ' not in token for token in this_df.token), "There is a multiword token, check if it has speaker tags."
    
    return filtered_df


def fix_gloss(gloss):
    return(str(gloss).replace('+','').replace('_',''))
 
def drop_errors(utt_data):
    
    cvt_lowercase = lambda s : str(s).lower()
    all_lowercase = list(map(cvt_lowercase, utt_data.gloss))
    
    assert len(all_lowercase) == utt_data.shape[0]
    
    utt_data['contains_error'] = ['xxx' in str(x) or 'yyy' in str(x) for x in all_lowercase]
    utt_data = utt_data.loc[~utt_data.contains_error]
    
    return utt_data



def clean_glosses(data):
        
    fill_punct_val = '.'
    
    # 8/1/21: changed for more punctuation types.
    punct_for_type = {
        'question':'?',
        'declarative':'.',
        'self interruption':'.',
        'interruption':'!',
        'trail off':'...',
        'interruption question':'?',
        'trail off question':'?',
        'imperative_emphatic':'!' 
    }
    
    data.gloss = [fix_gloss(x) for x in data.gloss]
    
    if config.verbose: print(data['type'].value_counts())
    
    # Cell 237
    data['punct'] = [punct_for_type[x] if x in punct_for_type else fill_punct_val
                        for x in data.type ]
    
    if config.verbose: print('Cell 238', data.iloc[0])
        
    # Cell 267
    
    assert all([x is not None for x in data.punct]) 
    assert not any(data.punct.isna())
    # Check to make sure that order of execution of following two lines doesn't matter
    
    data['speaker_code_simple'] = ['[CHI]' if x == 'CHI' else '[CGV]'
                                          for x in data.speaker_code]
    
    # Cell 268
    data = data.loc[[x is not None for x in data.punct]]
    data['gloss_with_punct'] = [x['speaker_code_simple'] + ' '+ x['gloss'].lower() + x['punct'] for x in data.to_dict('records')]
    
    return data



    