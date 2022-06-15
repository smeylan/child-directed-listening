import numpy as np
from src.utils import load_splits, load_models, sample_models_across_time, hyperparameter_utils, configuration, paths
config = configuration.Config()
    
def get_scores_across_models(test_idx, model_dicts, is_success):

    '''
    Get scores across a selection of models appropriate for an example figure. Looks at the results of run_beta_search to choose the best hyperparameter settings
    
    test_idx: utterance index
    which_models: selection of model specifications to run
    is_success: is the test_idx a communicative success (True) or communicative failure (False)

    '''

    scores_across_models = []
    success_ids, yyy_ids = [], []
   
    if is_success:
        success_ids = [test_idx]
    else:
        yyy_ids = [test_idx]
    
    all_tokens_phono = load_splits.load_phono()


    for model_dict in model_dicts:        

        # need to specify the test data so that it can load the appropriate model
        model_dict['task_name'] = 'analysis'
        model_dict['task_phase'] = 'eval' 
        model_dict['test_split'] = 'Providence'
        model_dict['test_dataset'] = 'all'  
        model_dict['n_samples'] = config.n_across_time
        model_dict['title'] = paths.get_file_identifier(model_dict)
        model_dict['examples_mode'] = True


        
        config.fail_on_lambda_edge =  False
        config.fail_on_beta_edge = False

        optimal_lambda_value = [hyperparameter_utils.get_optimal_hyperparameter_value(model_dict, 'lambda')]        
        if config.fail_on_lambda_edge:
            if optimal_lambda_value[0] >= config.lambda_high:
                raise ValueError('Lambda value is too high; examine the range for WFST scaling.')
            if optimal_lambda_value[0] <= config.lambda_low:
                raise ValueError('Lambda value is too low; examine the range for WFST Distance scaling.')

        
        optimal_beta_value = [hyperparameter_utils.get_optimal_hyperparameter_value(model_dict, 'beta')]
        if config.fail_on_beta_edge:
            if optimal_beta_value[0] >= config.beta_high:
                raise ValueError('Beta value is too high; examine the range for Levenshtein Distance scaling.')
            if optimal_beta_value[0] <= config.beta_low:
                raise ValueError('Beta value is too low; examine the range for Levenshtein Distance scaling.')

        this_model_dict = load_models.get_fitted_model_dict(model_dict)

        best_beta_scores = sample_models_across_time.successes_and_failures_across_time_per_model(0, success_ids, yyy_ids, this_model_dict, all_tokens_phono, optimal_beta_value[0], 'levdist')
        best_beta_scores['likelihood_type'] = 'levdist'
        best_beta_scores['model'] = model_dict['title']
        scores_across_models.append(best_beta_scores)

        best_lambda_scores = sample_models_across_time.successes_and_failures_across_time_per_model(0, success_ids, yyy_ids, this_model_dict, all_tokens_phono, optimal_lambda_value[0], 'wfst')
        best_lambda_scores['likelihood_type'] = 'wfst'
        best_lambda_scores['model'] = model_dict['title']
        scores_across_models.append(best_lambda_scores)
        

    return scores_across_models


