import os
import sys
from os.path import join, exists
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import copy

sys.path.append('.')
sys.path.append('src/.')
from utils import load_models, load_splits, data_cleaning, parsers, hyperparameter_utils, sample_models_across_time, configuration, paths
config = configuration.Config()


def use_child_specific_wfst(fitting_dict):
#     # we should only get the results of the child-specific WFST when the child and the training data are the same
    return((fitting_dict['training_split'] == 'Providence-Child') and (fitting_dict['training_dataset'] == fitting_dict['test_dataset']))  
    
def call_single_across_time_model(sample_dict, all_tokens_phono, this_model_dict):

    '''
        Load the best performing hyperparameter values for a given model and test dataset and run all eval data, saving the data by each 6 month time period
    '''       

    
    # re-use the lambda bounds for Gamma
    optimal_gamma_value = [hyperparameter_utils.get_optimal_hyperparameter_value(this_model_dict, 'gamma')]
    if config.fail_on_lambda_edge:
        if optimal_gamma_value[0] >= config.lambda_high:
            raise ValueError('Lambda value is too high for a child-specific dataset; examine the range for WFST scaling.')
        if optimal_gamma_value[0] <= config.lambda_low:
            raise ValueError('Lambda value is too low for a child-specific dataset; examine the range for WFST Distance scaling.')

    # Load the optimal beta and lambda
    optimal_lambda_value = [hyperparameter_utils.get_optimal_hyperparameter_value(this_model_dict, 'lambda')]
    if config.fail_on_lambda_edge:
        if optimal_lambda_value[0] >= config.lambda_high:
            raise ValueError('Lambda value is too high; examine the range for WFST scaling.')
        if optimal_lambda_value[0] <= config.lambda_low:
            raise ValueError('Lambda value is too low; examine the range for WFST Distance scaling.')

    
    optimal_beta_value = [hyperparameter_utils.get_optimal_hyperparameter_value(this_model_dict, 'beta')]
    if config.fail_on_beta_edge:
        if optimal_beta_value[0] >= config.beta_high:
            raise ValueError('Beta value is too high; examine the range for Levenshtein Distance scaling.')
        if optimal_beta_value[0] <= config.beta_low:
            raise ValueError('Beta value is too low; examine the range for Levenshtein Distance scaling.')


    ages = sorted(list(sample_dict.keys()))
   
    for idx, age_str in enumerate(ages):
        
        print('Processing age '+ age_str)
        
        age = float(age_str)
        
        percentage_done = idx / float(len(ages)) * 100
        
        if int(percentage_done) % 10 == 0: print(f'{percentage_done}%')
        
        this_pool = sample_dict[age_str]
        this_success_pool = this_pool['success']
        this_yyy_pool = this_pool['yyy']
        
        if (this_success_pool.shape[0] == 0) and (this_yyy_pool.shape[0] == 0): continue

        scores_output_path = paths.get_directory(this_model_dict)
        if not os.path.exists(scores_output_path):
            os.makedirs(scores_output_path)

        if use_child_specific_wfst(this_model_dict):

            # only evaluate the child-specific wfst in certain cases
            best_gamma_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, this_model_dict, all_tokens_phono, optimal_gamma_value[0], 'wfst-child')
            best_gamma_scores.to_pickle(join(scores_output_path, f'wfst-child_run_models_across_time_{age_str}.pkl'))
        else:
            best_gamma_scores = None
         
        best_beta_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, this_model_dict, all_tokens_phono, optimal_beta_value[0], 'levdist')        
        best_beta_scores.to_pickle(join(scores_output_path, f'levdist_run_models_across_time_{age_str}.pkl'))

        
        best_lambda_scores = sample_models_across_time.successes_and_failures_across_time_per_model(age, this_success_pool.utterance_id, this_yyy_pool.utterance_id, this_model_dict, all_tokens_phono, optimal_lambda_value[0], 'wfst')
        best_lambda_scores.to_pickle(join(scores_output_path, f'wfst_run_models_across_time_{age_str}.pkl'))


    return best_beta_scores, best_lambda_scores, best_gamma_scores
    
if __name__ == '__main__':
    
    start_time = str(datetime.today())
    
    parser = parsers.split_parser()
    parser.add_argument('--examples_mode', nargs = '?', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_model_args = vars(raw_args)    
    this_model_args['task_phase'] = 'eval'
    this_model_args['n_samples'] = config.n_across_time   
    print(this_model_args)
    
                                                                    
    all_phono = load_splits.load_phono()
    
    # this logic needs to be tested
    
    
    if (this_model_args['test_split'] == 'Providence') and (this_model_args['test_dataset'] == 'all'): 
        this_sample_dict = load_splits.load_sample_model_across_time_args()    

        import pdb
        pdb.set_trace()

        
    elif (this_model_args['test_split'] == 'Providence-Child'):
        #think about where this was in the notebooks        

        this_sample_dict = {}
        
        eval_samples = all_phono.loc[(all_phono.phase_child_sample == 'eval') & (all_phono.target_child_name == this_model_args['test_dataset'])]

        for age in [x for x in np.unique(eval_samples.year) if not np.isnan(x)]:

            eval_samples_age =  eval_samples.loc[eval_samples.year == age]


            success_utts = eval_samples_age[eval_samples_age.success_token][['utterance_id']].drop_duplicates()
            failure_utts = eval_samples_age[eval_samples_age.yyy_token][['utterance_id']].drop_duplicates()

            if success_utts.shape[0] > config.n_across_time:
                success_utts = success_utts.iloc[0:config.n_across_time]

            if failure_utts.shape[0] > config.n_across_time:
                failure_utts = failure_utts.iloc[0:config.n_across_time]

            rdict = {'success': success_utts, 'yyy': failure_utts}

            this_sample_dict[str(age)] = copy.copy(rdict)

    else:
        raise NotImplementedError


    this_model_dict = load_models.get_fitted_model_dict(this_model_args)

     
    best_beta_scores, best_lambda_scores, best_gamma_scores = call_single_across_time_model(this_sample_dict, all_phono, this_model_dict)
    
    print(f'Computations complete for model:')
    print(this_model_args)
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
