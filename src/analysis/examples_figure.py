import numpy as np
from src.utils import transformers_bert_completions, load_splits, load_models, sample_across_models, hyperparameter_utils, configuration
config = configuration.Config()
    
def get_scores_across_models(test_idx, which_models, is_success):

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

    for args_extract in which_models:

        model_dict = load_models.get_model_dict(*args_extract)
        
        config.fail_on_lambda_edge =  False
        config.fail_on_beta_edge = False

        optimal_lambda_value = [hyperparameter_utils.get_optimal_hyperparameter_value(*args_extract, 'lambda')]        
        if config.fail_on_lambda_edge:
            if optimal_lambda_value[0] >= config.lambda_high:
                raise ValueError('Lambda value is too high; examine the range for WFST scaling.')
            if optimal_lambda_value[0] <= config.lambda_low:
                raise ValueError('Lambda value is too low; examine the range for WFST Distance scaling.')

        
        optimal_beta_value = [hyperparameter_utils.get_optimal_hyperparameter_value(*args_extract, 'beta')]
        if config.fail_on_beta_edge:
            if optimal_beta_value[0] >= config.beta_high:
                raise ValueError('Beta value is too high; examine the range for Levenshtein Distance scaling.')
            if optimal_beta_value[0] <= config.beta_low:
                raise ValueError('Beta value is too low; examine the range for Levenshtein Distance scaling.')

        this_scoring = sample_across_models.sample_across_models(success_ids, yyy_ids, model_dict, optimal_beta_value, optimal_lambda_value, examples_mode = True, all_tokens_phono = all_tokens_phono)

        scores_across_models.append(this_scoring)

    return scores_across_models