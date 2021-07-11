

from utils import transformers_bert_completions, load_splits, load_models
from utils_model_sampling import sample_across_models, beta_utils

from yyy_analysis import examples_figures

def get_example_model_ids():
    
    # CDL + Context +/- 20 is needed
    # BERT + Context +/- 20 is needed
    # Childes on train data.

    # How to load properly with sample across models?
    which_models = [
        'all/all/no_tags/20_context/childes',
        'all/all/no_tags/20_context/bert',
        'all/all/no_tags/0_context/data_unigram',
    ]
    
    return which_models
    
    
def get_scores_across_models(test_idx):

    all_models = load_models.get_model_dict()
    
    which_models = get_example_model_ids()

    scores_across_models = []

    for model_id in which_models:

        print(model_id)

        args_extract = model_id.split('/')
        this_split, this_dataset = args_extract[0], args_extract[1]
        this_model_type = args_extract[-1]

        this_model_dict = all_models[model_id]

        optimal_beta = beta_utils.get_optimal_beta_value(this_split, this_dataset, this_model_dict, this_model_type)
        this_scoring = sample_across_models.sample_across_models([test_idx], 
            success_utts, yyy_utts, all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab, beta_values=[optimal_beta])

        scores_across_models.append(this_scoring)

    return scores_across_models

