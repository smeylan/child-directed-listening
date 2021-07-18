
import os
from os.path import join, exists

import copy
from utils import load_models, transformers_bert_completions, load_csvs, unigram, load_splits
from utils_model_sampling import beta_utils

from collections import defaultdict
import numpy as np

import glob


def assemble_scores_no_order():
    """
    Assumes order of the the model vs age loop doesn't matter.
    """
    
    this_load_args = load_models.gen_all_model_args()
    
    score_store = []
    
    for split, dataset, tags, context, model_type in this_load_args:
       
        this_beta_folder = beta_utils.load_beta_folder(split, dataset, tags, context, model_type)

        age_paths = glob.glob(join(this_beta_folder, 'run_models_across_time_*.csv'))
         
        for this_data_path in age_paths:
            
            print(this_data_path)
            
            data_df = load_csvs.load_csv_with_lists(this_data_path)
            
            # print(f'\t{this_data_path}')
            # print(f'\tthis shape {data_df.shape[0]}')
            
            score_store.append(data_df)
                      
    return score_store
    
    
def assemble_across_time_scores():
    
    """
    
    Follows the convention of the original code -- need to work on this more.
     
    Assemble the "score store" across models that was present in the original function
        and is used for visualizations.
    Outer loop is by age.
    Inner loop is by model, for that pool.
    Note that different splits have different samples of data.
    
    Doesn't work yet -- abandoning because it's probably unneeded
    """
    
    this_load_args = load_models.gen_all_model_args()
    
    # For now, analyze whichever ages are available in the sample.
    # Need to be careful when doing visualizations in yyy later.
    
    # First, access each pool for their samples
    
    age2models = defaultdict(list)
    
    for split, dataset, _, _, _ in this_load_args:
        # Not just successes! What else to load?
        this_sample_pool = load_splits.load_sample_successes('models_across_time', split, dataset)
        
        this_ages = np.unique(this_sample_pool.target_child_age)
        age2models[age].append((split, dataset)) # Not sure if age is a float or int, be careful
    
    all_ages = sorted(list(age2models.keys()))
    
    
    # Then, sort all of the model calls by age
    # Age -> model -> scores nesting
    
    score_store = []
   
    for age in all_ages:
        for split, dataset, tags, context, model_type in this_load_args:
            this_beta_folder = beta_utils.load_beta_folder(split, dataset, tags, context, model_type)

            this_data_path = join(this_beta_folder, 'run_models_across_time_{age}.csv')
            data_df = load_csvs.load_csv_with_lists(this_data_path)
            score_store.append(data_df)

    return score_store

def successes_and_failures_across_time_per_model(age, utts, model, all_tokens_phono, beta_value):
    """
    model = a dict of a model like that in the yyy analysis 
    vocab is only invoked for unigram, which correspond to original yyy analysis.
    
    Unlike original code assume that utts = the sample of utts_with_ages, not the whole dataframe
    """
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    
    
    print('Running model '+model['title']+f'... at age {age}')
    
    selected_success_utts = utts.loc[(utts.set == 'success') 
            & (utts.year == age)]
    
    selected_yyy_utts = utts.loc[(utts.set == 'failure') 
            & (utts.year == age)]
    
    # Note that if the age doesn't yield both successes and failures,
    # then one of the dataframes can be empty
    # causing runtime error -> program doesn't run to completion.
    # This is very unlikely for large samples, but potentially causes runtime errors in the middle of running.
    
    if model['type'] == 'BERT':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures(
            all_tokens_phono, selected_success_utts.utterance_id, 
            selected_yyy_utts.utterance_id, **model['kwargs'])

    elif model['type'] == 'unigram':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures_unigram_model(
            all_tokens_phono, selected_success_utts.utterance_id, 
            selected_yyy_utts.utterance_id, **model['kwargs'])

    edit_distances_for_age_interval = transformers_bert_completions.get_edit_distance_matrix(all_tokens_phono, 
        priors_for_age_interval, initial_vocab, cmu_in_initial_vocab)            

    if model['type'] == 'BERT':
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
            edit_distances_for_age_interval, initial_vocab, beta_value = beta_value)
    elif model['type'] == 'unigram':
        # special unigram hack
        this_bert_token_ids = unigram.get_sample_bert_token_ids('models_across_time')
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, initial_vocab, this_bert_token_ids, beta_value = beta_value)


    posteriors_for_age_interval['scores']['model'] = model['title']
    posteriors_for_age_interval['scores']['age'] = age
    
    return copy.deepcopy(posteriors_for_age_interval['scores'])
