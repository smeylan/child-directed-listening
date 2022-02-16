import os
from os.path import join, exists
import pandas as pd
import numpy as np
from src.utils import configuration, load_models, split_gen
config = configuration.Config()


def get_hyperparameter_search_values(hyperparam):

    '''
    Generate the range of hyperparameter values given the parameters that are in the config file

    Args: 
    hyperparam: 'lambda' or 'beta'

    Return:
    A range of values for the specified hyperparameter
    '''
    
    low = getattr(config, hyperparam+'_low')
    high = getattr(config, hyperparam+'_high')
    num_values = getattr(config, hyperparam+'_num_values')
    
    hyperparameter_samples = np.arange(low, high, (high - low) / num_values)
    
    return hyperparameter_samples    


def get_optimal_hyperparameter_value_with_dict(split, dataset, model_dict, model_type, hyperparameter):

    '''
    A convenience wrapper to be able to call get_optimal_hyperparameter_value with a model_dict in some cases    
    '''
    
    return get_optimal_hyperparameter_value(split, dataset, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type, hyperparameter)
    

def load_hyperparameter_folder(split, dataset, tags, context, model_type):

    '''
    Load hyperparameter results (both lambda and beta) of run_beta_search.py for a single model

    split: If the model is a fine-tuned BERT model, is it trained on all CHILDES data, young children, or old chilren
    dataset: what dataset should be evaluated?
    tags: If the model is a fine-tuned BERT model, does it contain tags
    context: How many utterances before and after the target token
    model_type: model label, choose 'childes' for fine-tuned BERT, 'adult' for off the shelf BERT, 'flat_unigram' for UniformPrior, 'data_unigram' for CHILDES-unigram
    hyperparameter folder

    Return
    The path to the hyperparameter folder

    '''     

    folder = split_gen.get_split_folder(split, dataset, config.scores_dir)
    this_title = load_models.query_model_title(split, dataset, tags, context, model_type)
    exp_path = join(folder, this_title.replace(' ', '_'))

    if not exists(exp_path):
        os.makedirs(exp_path)
    
    return exp_path    

    
def get_optimal_hyperparameter_value(split, dataset, tags, context, model_type, hyperparameter):

    '''
    Get the best hyperparameter value from the results of run_beta_search.py

    split: If the model is a fine-tuned BERT model, is it trained on all CHILDES data, young children, or old chilren
    dataset: what dataset should be evaluated?
    tags: If the model is a fine-tuned BERT model, does it contain tags
    context: How many utterances before and after the target token
    model_type: model label, choose 'childes' for fine-tuned BERT, 'adult' for off the shelf BERT, 'flat_unigram' for UniformPrior, 'data_unigram' for CHILDES-unigram
    hyperparameter folder
    hyperparameter: 'beta' or 'lambda'

    Return
    The best-scoring hyperparameter value 

    '''     

    exp_model_path = load_hyperparameter_folder(split, dataset, tags, context, model_type)
    
    if hyperparameter == 'beta':     
        n_hyperparameter = config.n_beta    
    elif hyperparameter == 'lambda':     
        n_hyperparameter = config.n_lambda       
    
    this_hyperparameter_results  =  pd.read_csv(join(exp_model_path, hyperparameter+f'_search_results_{n_hyperparameter}.csv'))
    
    # Need to argmax for beta_value, given the posterior surprisal
    list_hyperparameter_results = list(this_hyperparameter_results[hyperparameter+'_value'])
    list_surp = list(this_hyperparameter_results['posterior_surprisal'])
    
    argmin_hyperparameter = np.argmin(list_surp)
    best_hyperparameter = list_hyperparameter_results[argmin_hyperparameter]

    return best_hyperparameter
    
    
    
    
    
