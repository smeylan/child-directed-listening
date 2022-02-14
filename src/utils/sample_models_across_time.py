import os
from os.path import join, exists
import copy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import pickle5 as pickle

from src.utils import load_models, transformers_bert_completions, load_splits, likelihoods, hyperparameter_utils, configuration 
config = configuration.Config()

def assemble_scores_no_order(hyperparameter_set):
    """
    Assumes order of the the model vs age loop doesn't matter.
    """
    
    # hyperparameter_set: wfst or levdist
    
    this_load_args = load_models.gen_all_model_args()
    
    score_store = []
    
    for split, dataset, tags, context, model_type in this_load_args:
       
        this_hyperparameter_folder = hyperparameter_utils.load_hyperparameter_folder(split, dataset, tags, context, model_type)

        search_string = join(this_hyperparameter_folder, hyperparameter_set+'_run_models_across_time_*.pkl')
        age_paths = glob.glob(search_string)
        
        for this_data_path in age_paths:
            
            #data_df = pd.read_pickle(this_data_path)
            with open(this_data_path, "rb") as fh:
                data_df = pickle.load(fh)
            
            score_store.append(data_df)
                      
    return score_store



def successes_and_failures_across_time_per_model(age, success_ids, yyy_ids, model, all_tokens_phono, beta_value, examples_mode, likelihood_type):
    """
    model = a dict of a model like that in the yyy analysis 
    vocab is only invoked for unigram, which correspond to original yyy analysis.
    beta_value: generic name for beta or lambda (really a scaling value)
    
    Unlike original code assume that utts = the sample of utts_with_ages, not the whole dataframe
    """
    
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab  = load_models.get_initial_vocab_info()
    
    print('Running model '+model['title']+f'... at age {age}')
    
    # Note that if the age doesn't yield both successes and failures,
    # then one of the dataframes can be empty
    # causing runtime error -> program doesn't run to completion.
    # This is very unlikely for large samples, but potentially causes runtime errors in the middle of running.
    
    if model['type'] == 'BERT':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])

    elif model['type'] == 'unigram':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures_unigram_model(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])

    # run the best model    
    if likelihood_type == 'wfst':
        likelihood_matrix, ipa = likelihoods.get_wfst_distance_matrix(all_tokens_phono, priors_for_age_interval, initial_vocab,  cmu_in_initial_vocab, config.fst_path, config.fst_sym_path)
        likelihood_matrix = -1 * np.log(likelihood_matrix + 10**-20) # yielding a surprisal
    elif likelihood_type == 'levdist':
        likelihood_matrix = likelihoods.get_edit_distance_matrix(all_tokens_phono, 
            priors_for_age_interval, cmu_in_initial_vocab)            

    # likelihood_matrix has all pronunciation variants     
    likelihood_matrix = likelihoods.reduce_duplicates(likelihood_matrix, cmu_in_initial_vocab, initial_vocab, 'min', cmu_indices_for_initial_vocab)


    if model['type'] == 'BERT':
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
            likelihood_matrix, initial_vocab, scaling_value = beta_value, examples_mode = examples_mode)
    elif model['type'] == 'unigram':
        # special unigram hack
        this_bert_token_ids = all_tokens_phono.loc[all_tokens_phono.partition.isin(('success','yyy'))].bert_token_id

        #this_bert_token_ids = unigram.get_sample_bert_token_ids()
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, likelihood_matrix, initial_vocab, this_bert_token_ids, scaling_value = beta_value, examples_mode = examples_mode)

    posteriors_for_age_interval['scores']['model'] = model['title']
    posteriors_for_age_interval['scores']['age'] = age


    
    return copy.deepcopy(posteriors_for_age_interval['scores'])
   
