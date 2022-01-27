
from utils import transformers_bert_completions, load_splits, load_models
from utils_model_sampling import sample_across_models, hyperparameter_utils

    
def get_scores_across_models(test_idx, which_models, is_success):

    scores_across_models = []
    success_ids, yyy_ids = [], []
   
    if is_success:
        success_ids = [test_idx]
    else:
        yyy_ids = [test_idx]

    
    all_tokens_phono = load_splits.load_phono()


    for args_extract in which_models:

        model_dict = load_models.get_model_dict(*args_extract)
        
        optimal_beta_value = [hyperparameter_utils.get_optimal_hyperparameter_value(*args_extract, 'beta')]
        optimal_lambda_value = [hyperparameter_utils.get_optimal_hyperparameter_value(*args_extract, 'lambda')]
        
        this_scoring = sample_across_models.sample_across_models(success_ids, yyy_ids, model_dict, optimal_beta_value, optimal_lambda_value, examples_mode = True, all_tokens_phono = all_tokens_phono)

        scores_across_models.append(this_scoring)

    return scores_across_models

