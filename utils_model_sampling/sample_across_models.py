
import copy
import pandas as pd

from utils import load_models, transformers_bert_completions, unigram

def sample_across_models(utterance_ids, model, eval_data_dict, beta_values):
    '''
        Top-level method to sample all models for a set of communicative successes and failures. Allows for adjusting the beta value
        
        Note: model = model dictionary from the load models functions, not a huggingface model alone.
        
        >>> Notes of caution from original code:
        
        initial_vocab: word types corresponding to the softmask mask !!! potentially brittle:  different softmax masks per model 
        cmu_in_initial_vocab: cmu pronunciations for the initial vocabulary !!! potentially brittle interaction with initial vocab       
        beta values: list of beta values over which to iterate
        
        We decided it was best to use same initial_vocab and cmu_in_initial_vocab for everything (on CHILDES data) because it prevents changing the quantitative meaning of the softmax scores.

    '''
    
    # Note: utterance_ids can't be a Dataframe or an empty sample will result
     
    all_tokens_phono = eval_data_dict['phono']
    success_utts = eval_data_dict['success_utts']
    yyy_utts = eval_data_dict['yyy_utts']
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()

    print('Running model '+model['title']+'...')
    
    selected_success_utts = success_utts.loc[success_utts.utterance_id.isin(utterance_ids)]
    selected_yyy_utts = yyy_utts.loc[yyy_utts.utterance_id.isin(utterance_ids)] 

    # get the priors
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

    score_store_single_model = []
    
    for beta_value in beta_values:

        # get the posteriors        
        if model['type'] == 'BERT':
            posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
                edit_distances_for_age_interval, initial_vocab, None, beta_value)

        elif model['type'] == 'unigram':
            # special unigram hack
            this_bert_token_ids = unigram.get_sample_bert_token_ids('beta')
            
            posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, 
                initial_vocab, this_bert_token_ids, beta_value)
            print('If possible compare the bert_token_id in sample_across_models to the bert_token_id in one of the other scores sets from bert.')
            
        posteriors_for_age_interval['scores']['beta_value'] = beta_value
        posteriors_for_age_interval['scores']['model'] = model['title']
    
        score_store_single_model.append(copy.deepcopy(posteriors_for_age_interval['scores']))
        
    all_scores = pd.concat(score_store_single_model)
    return all_scores 