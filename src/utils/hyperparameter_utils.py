import os
from os.path import join, exists
import pandas as pd
import numpy as np
import copy
from src.utils import configuration, load_models, split_gen, paths
config = configuration.Config()


def get_hyperparameter_search_values(hyperparam):

    '''
    Generate the range of hyperparameter values given the parameters that are in the config file

    Args: 
    hyperparam: 'lambda', 'gamma' or 'beta'

    Return:
    A range of values for the specified hyperparameter
    '''
    
    low = getattr(config, hyperparam+'_low')
    high = getattr(config, hyperparam+'_high')
    num_values = getattr(config, hyperparam+'_num_values')
    
    hyperparameter_samples = np.arange(low, high, (high - low) / num_values)
    
    return hyperparameter_samples    


def get_optimal_hyperparameter_value(this_model_dict, hyperparameter):

    '''
        Get the best hyperparameter value for a given model x test dataset
    '''


    fitted_model_dict = copy.copy(this_model_dict)
    fitted_model_dict['task_phase'] = 'fit'

    fitted_model_path = paths.get_directory(fitted_model_dict)
    

    if hyperparameter == 'beta':     
        n_hyperparameter = config.n_beta    
    elif hyperparameter in ['lambda','gamma']:     
        n_hyperparameter = config.n_lambda       
    
    this_hyperparameter_results  =  pd.read_csv(join(fitted_model_path, hyperparameter+f'_search_results_{n_hyperparameter}.csv'))
    
    # # Need to argmax for beta_value, given the posterior surprisal
    list_hyperparameter_results = list(this_hyperparameter_results[hyperparameter+'_value'])
    list_surp = list(this_hyperparameter_results['posterior_surprisal'])
    
    argmin_hyperparameter = np.argmin(list_surp)
    best_hyperparameter = list_hyperparameter_results[argmin_hyperparameter]

    return best_hyperparameter


    
    
    
    
    
    
