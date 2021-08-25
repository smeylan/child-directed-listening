
from utils import transformers_bert_completions, load_splits, load_models
from utils_model_sampling import sample_across_models, beta_utils

    
def get_scores_across_models(test_idx, which_models, is_success):

    scores_across_models = []
    
    success_utts, yyy_utts = [], []
    
    all_tokens_phono = load_splits.load_phono()
    
    if is_success:
        success_utts = [test_idx]
    else:
        yyy_utts = [test_idx]

    for args_extract in which_models:

        model_dict = load_models.get_model_dict(*args_extract)
        
        optimal_beta = beta_utils.get_optimal_beta_value(*args_extract)
        
        this_scoring = sample_across_models.sample_across_models([test_idx], 
            success_utts, yyy_utts, all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab, beta_values=[optimal_beta], examples_mode = True)

        scores_across_models.append(this_scoring)

    return scores_across_models

