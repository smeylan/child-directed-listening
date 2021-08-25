
## To generate data for early Section 2 analyses in the original yyy notebook.

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

from utils import load_models

import configuration
config = configuration.Config()

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
    
    if config.verbose:
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
    
    if config.verbose:
        print(yyy_utts.shape)
        
    success_utts = failures_per_utt.loc[(failures_per_utt.num_xxx == 0) &  (failures_per_utt.num_yyy == 0)]
    
    if config.verbose:
        print(success_utts.shape)
    
    tokens_from_errorless_utts = chi_phono.loc[chi_phono.utterance_id.isin(success_utts.utterance_id)]
    #exclude un-transcribed tokens and syllabically transcribed tokens
    excludes = ['*','(.)','(..)', '(...)','(....)','(.....)']
    tokens_from_errorless_utts = tokens_from_errorless_utts.loc[~(tokens_from_errorless_utts.actual_phonology.isin(excludes) |
        tokens_from_errorless_utts.model_phonology.isin(excludes))]
    
    if config.verbose:
        print(tokens_from_errorless_utts.shape)
        print()
        print(tokens_from_errorless_utts.actual_phonology)
        
    chi_phono = chi_phono.loc[chi_phono.gloss != "xxx"] # Part of Section 2 but the last place chi_phono is edited.
    
    if config.verbose: print(chi_phono.shape)
    
    return chi_phono

