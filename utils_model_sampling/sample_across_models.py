
import copy
import pandas as pd

from utils import load_models, transformers_bert_completions, unigram, load_splits

import configuration
config = configuration.Config()

def sample_across_models(success_ids, yyy_ids, model, beta_values, examples_mode = False):
    '''
        Top-level method to sample all models for a set of communicative successes and failures. Allows for adjusting the beta value
        
        Note: model = model dictionary from the load models functions, not a huggingface model alone.
        
        >>> Notes of caution from original code:
        
        initial_vocab: word types corresponding to the softmask mask !!! potentially brittle:  different softmax masks per model 
        cmu_in_initial_vocab: cmu pronunciations for the initial vocabulary !!! potentially brittle interaction with initial vocab
        
        We decided it was best to use same initial_vocab and cmu_in_initial_vocab for everything (on CHILDES data) because it prevents changing the quantitative meaning of the softmax scores.
        
        examples_mode = whether or not to retain highest probability word-related information, False to save memory

    '''
    
    # Note: utterance_ids can't be a Dataframe or an empty sample will result
     
    all_tokens_phono = load_splits.load_phono()
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()

    print('Running model '+model['title']+'...')

    # get the priors
    if model['type'] == 'BERT':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])

    elif model['type'] == 'unigram':
        priors_for_age_interval = transformers_bert_completions.compare_successes_failures_unigram_model(
            all_tokens_phono, success_ids, 
            yyy_ids, **model['kwargs'])
      
    edit_distances_for_age_interval = transformers_bert_completions.get_edit_distance_matrix(all_tokens_phono, 
        priors_for_age_interval, initial_vocab, cmu_in_initial_vocab)         

    score_store_single_model = []
    
    for idx, beta_value in enumerate(beta_values):
        
        print(f'Processing beta value {idx + 1} of {config.num_values}')

        # get the posteriors        
        if model['type'] == 'BERT':
            posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
                edit_distances_for_age_interval, initial_vocab, None, beta_value, examples_mode = examples_mode)

        elif model['type'] == 'unigram':
            # special unigram hack
            this_bert_token_ids = unigram.get_sample_bert_token_ids()
            
            posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, 
                initial_vocab, this_bert_token_ids, beta_value, examples_mode = examples_mode)
            print('If possible compare the bert_token_id in sample_across_models to the bert_token_id in one of the other scores sets from bert.')
            
        posteriors_for_age_interval['scores']['beta_value'] = beta_value
        posteriors_for_age_interval['scores']['model'] = model['title']
        
        posteriors_for_age_interval['scores'].astype({'beta_value' : 'float16'})
        this_score = copy.deepcopy(posteriors_for_age_interval['scores'])
        
        score_store_single_model.append(this_score)
        
    all_scores = pd.concat(score_store_single_model)
    
    return all_scores 