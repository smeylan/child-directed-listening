## To generate data for yyy code in analysis.

import rpy2.robjects.lib.ggplot2 as ggplot2
import childespy
import numpy as np
import os
import imp
import pandas as pd
import transformers
import torch
import re
import unicodedata
import scipy.stats
import copy
from string import punctuation

from util import load_models
import transfomers_bert_completions # Check if this import is possible...

def count_transmission_errors(utt_vector, error_codes):
    return(np.sum([x in error_codes for x in  utt_vector]))

def get_communication_failures_data(verbose = False):
    
    """
    This code gives the input to the section Prevalence of Communication Failures in the notebook analysis.
    It also gives the raw input to be cleaned for Sections 3 and onward of the original yyy notebook.
    """
    
    # Communicative success: how many no-xxx, no-yyy child  utterances are in Providence? 
    # Communicative failures: how many one-yyy, no-xxx child utterances are in Providence?
    # Subset to instances that are monosyllabic later
    
    pvd_idx = childespy.get_sql_query('select * from corpus where name = "Providence"').iloc[0]['id']
    
    phono_glosses = childespy.get_sql_query('select gloss, target_child_name, target_child_age, \
    speaker_code, actual_phonology, model_phonology, transcript_id, utterance_id, \
    token_order, corpus_name, collection_name, language from token where \
    actual_phonology != "" and model_phonology != "" and collection_name = "Eng-NA" \
    and corpus_id = '+str(pvd_idx) ,
        db_version = "2020.1")
    
    if verbose:
        print(phono_glosses.corpus_name.value_counts())
        print()
        print(phono_glosses.loc[phono_glosses.gloss == 'xxx'].actual_phonology.value_counts())
        print(phono_glosses.loc[phono_glosses.gloss == 'yyy'].actual_phonology.value_counts())
        
    chi_phono = phono_glosses.loc[(phono_glosses.speaker_code == 'CHI') & 
    (phono_glosses.target_child_age < (365*5))]
    
    xxxs_per_utt = chi_phono.groupby('utterance_id').gloss.agg(
        lambda x: count_transmission_errors(x, ['xxx'])).reset_index()
    xxxs_per_utt.columns = ['utterance_id', 'num_xxx']
    yyys_per_utt = chi_phono.groupby('utterance_id').gloss.agg(
        lambda x: count_transmission_errors(x, ['yyy'])).reset_index()
    
    yyys_per_utt.columns = ['utterance_id', 'num_yyy']
    
    failures_per_utt = xxxs_per_utt.merge(yyys_per_utt)

    yyy_utts = failures_per_utt.loc[(failures_per_utt.num_xxx == 0) &  (failures_per_utt.num_yyy == 1)]
    
    if verbose:
        print(yyy_utts.shape)
        
    success_utts = failures_per_utt.loc[(failures_per_utt.num_xxx == 0) &  (failures_per_utt.num_yyy == 0)]
    
    if verbose:
        print(success_utts.shape)
    
    tokens_from_errorless_utts = chi_phono.loc[chi_phono.utterance_id.isin(success_utts.utterance_id)]
    #exclude un-transcribed tokens and syllabically transcribed tokens
    excludes = ['*','(.)','(..)', '(...)','(....)','(.....)']
    tokens_from_errorless_utts = tokens_from_errorless_utts.loc[~(tokens_from_errorless_utts.actual_phonology.isin(excludes) |
        tokens_from_errorless_utts.model_phonology.isin(excludes))]
    
    if verbose:
        print(tokens_from_errorless_utts.shape)
        print()
        print(tokens_from_errorless_utts.actual_phonology)
        
    chi_phono = chi_phono.loc[chi_phono.gloss != "xxx"] # Part of Section 2 but the last place chi_phono is edited.
    
    if verbose: print(chi_phono.shape)
    
    return chi_phono


def query_bert_providence_data(regenerate = False):
    
    """
    This data is the original evaluation set of yyy analysis
        (and will be for the "all" split, as well as the "old" and "young" split, once this data is also split on age.)
    However, this data is the original unsplit dataset for the child analysis.
    """
    
    # Get the index of the Providence corpus
    pvd_idx = childespy.get_sql_query('select * from corpus where name = "Providence"').iloc[0]['id']
    
    # Load utterances from the Providence corpus from childs-db
    if regenerate:
        utt_glosses = childespy.get_sql_query('select gloss, transcript_id, id, \
        utterance_order, speaker_code, target_child_name, target_child_age, type from utterance where corpus_id = '+str(pvd_idx) ,
            db_version = "2020.1")
        utt_glosses.to_csv('csv/pvd_utt_glosses.csv', index=False)
    else: 
        utt_glosses = pd.read_csv('csv/pvd_utt_glosses.csv')
    
    return utt_glosses

    
def query_clean_bert_data(regenerate = False, verbose = False):
    
    utt_data = query_bert_data(regenerate)
    utt_data = clean_glosses(utt_data, '.')
    
    if verbose: print(utt_glosses[utt_glosses.id == 17280964])
    
    return utt_data


def inflate(row):
    tokens = initial_tokenizer.tokenize(row['gloss_with_punct'])
    return(pd.DataFrame({'token':tokens, 'id':row['id']}) )


def inflate_glosses(utt_data, regenerate = False)


    print('inflate_glosses, get_analysis_data.py')
    print('Need to change the data so that it can save for child-specific data splits')
    
    # build a dataframe of tokens 
    # this is slow, because tokenization is slow

    if regenerate:
        all_tokens = pd.concat([inflate(x) for x in utt_data.to_dict('records')])
        all_tokens = all_tokens.merge(utt_data)
        all_tokens.to_csv('csv/pvd_utt_glosses_inflated.csv')

    else:
        all_tokens = pd.read_csv('csv/pvd_utt_glosses_inflated.csv', na_filter=False)
    
    return all_tokens
    
    
def build_bert_vocab(data, verbose = False, root_dir = '..'):
    
    cmu_2syl_inchildes = load_models.get_cmu_dict_info(root_dir)

    # tokenize with the most extensive tokenizer, which is the one used for model #2

    initial_tokenizer = load_models.get_meylan_original_model(with_tags = True, root_dir)['tokenizer']

    initial_tokenizer.add_tokens(['yyy','xxx']) #must maintain xxx and yyy for alignment,
    # otherwwise, BERT tokenizer will try to separate these into x #x and #x and y #y #y
    inital_vocab_mask, initial_vocab = transfomers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes.word)

    # confirm yyy treated as a separate character
    assert initial_tokenizer.tokenize('this is a yyy.') == ['this', 'is', 'a', 'yyy', '.']

    cmu_in_initial_vocab = cmu_2syl_inchildes.loc[cmu_2syl_inchildes.word.isin(initial_vocab)]

    if verbose: print(cmu_in_initial_vocab.shape)
        
    data_tokens = inflate_glosses(data)
    
    
    if verbose: print(data_tokens.iloc[0:10])
    
    # Assign a token_id (integer in the BERT vocabulary). 
    # Because these are from the tokenized utterances, there is no correpsondence 
    # with childes-db token ids
    data_tokens['token_id'] = initial_tokenizer.convert_tokens_to_ids(data_tokens['token'])
    # assigns utterances a 0-indexed index column
    data_tokens['seq_utt_id'] = data_tokens['id'].astype('category').cat.codes


def get_bert_data(cmu_dict_root_dir = '..', regenerate = False, verbose = False):
    
    clean_data = query_clean_bert_data(regenerate, verbose)
    clean_vocab = build_bert_vocab(clean_data, cmu_dict_root_dir)
    
    ## You are currently at: ### Add back IPA, syllable structure, and child ages for child productions
    


