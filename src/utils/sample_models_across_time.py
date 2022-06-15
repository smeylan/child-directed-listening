import os
from os.path import join, exists
import copy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import pickle5 as pickle

from src.utils import load_models, transformers_bert_completions, load_splits, likelihoods, hyperparameter_utils, configuration, paths 
config = configuration.Config()

def assemble_scores_no_order(hyperparameter_set):
    
    """
    Load all of the non_child models for a given hyperparameter
    """
    
    model_args = finetune_models = load_models.gen_finetune_model_args() + load_models.gen_shelf_model_args() + load_models.gen_unigram_args() 

    score_store = []
    
    for model_arg in model_args:

        model_arg['task_name'] = 'analysis'
        model_arg['task_phase'] = 'eval' 
        model_arg['test_split'] = 'Providence'
        model_arg['test_dataset'] = 'all'  
        model_arg['n_samples'] = config.n_across_time

        
        # loading from 
        results_path = paths.get_directory(model_arg)    
        search_string = join(results_path, hyperparameter_set+'_run_models_across_time_*.pkl')
        print('Searching '+search_string)
        age_paths = glob.glob(search_string)
        
        for this_data_path in age_paths:
            
            #data_df = pd.read_pickle(this_data_path)
            with open(this_data_path, "rb") as fh:
                data_df = pickle.load(fh)
                data_df['training_split'] = model_arg['training_split']
                data_df['training_dataset'] = model_arg['training_dataset']
                data_df['test_split'] = model_arg['test_split']
                data_df['test_dataset'] = model_arg['test_dataset']
                data_df['model_type'] = model_arg['model_type']
            

                data_df['split'] = data_df.training_split + '_' + data_df.training_dataset
                data_df['model'] = paths.get_file_identifier(model_arg)


            score_store.append(data_df)
                      
    return score_store



def successes_and_failures_across_time_per_model(age, success_ids, yyy_ids, model, all_tokens_phono, beta_value, likelihood_type):
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
    
    if model['model_type'] == 'BERT':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])

    elif model['model_type'] in ['data_unigram', 'flat_unigram']:
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures_unigram_model(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])
    else:
        raise ValueError('model_type not recognized!')

    # run the best model    
    if likelihood_type == 'wfst':
        child_general_fst_path = os.path.join(config.project_root,  config.fst_path)
        child_general_sym_path = os.path.join(config.project_root,  config.fst_sym_path)

        likelihood_matrix, ipa = likelihoods.get_wfst_distance_matrix(all_tokens_phono, priors_for_age_interval, initial_vocab,  cmu_in_initial_vocab, child_general_fst_path, child_general_sym_path)
        likelihood_matrix = -1 * np.log(likelihood_matrix + 10**-20) # yielding a surprisal
    
    elif likelihood_type == 'wfst-child':
        child_specific_fst_path = os.path.join(config.project_root,  model['training_dataset']+'-1.txt')
        child_specific_sym_path = os.path.join(config.project_root,  config.fst_sym_path)

        likelihood_matrix, ipa = likelihoods.get_wfst_distance_matrix(all_tokens_phono, priors_for_age_interval, initial_vocab,  cmu_in_initial_vocab, child_specific_fst_path, child_specific_sym_path)

        likelihood_matrix = -1 * np.log(likelihood_matrix + 10**-20) # yielding a surprisal
    elif likelihood_type == 'levdist':
        likelihood_matrix = likelihoods.get_edit_distance_matrix(all_tokens_phono, 
            priors_for_age_interval, cmu_in_initial_vocab)            
    else:
        raise ValueError('Likelihood not recognized!')

    # likelihood_matrix has all pronunciation variants     
    likelihood_matrix = likelihoods.reduce_duplicates(likelihood_matrix, cmu_in_initial_vocab, initial_vocab, 'min', cmu_indices_for_initial_vocab)


    if model['model_type'] == 'BERT':
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
            likelihood_matrix, initial_vocab, scaling_value = beta_value, examples_mode = model['examples_mode'])
    elif model['model_type'] in ['data_unigram', 'flat_unigram']:
        # special unigram hack
        this_bert_token_ids = all_tokens_phono.loc[all_tokens_phono.partition.isin(('success','yyy'))].bert_token_id

        #this_bert_token_ids = unigram.get_sample_bert_token_ids()
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, likelihood_matrix, initial_vocab, this_bert_token_ids, scaling_value = beta_value, examples_mode = model['examples_mode'])
    else:
        raise ValueError('model_type not recognized!')

    posteriors_for_age_interval['scores']['age'] = age

    return copy.deepcopy(posteriors_for_age_interval['scores'])
   
