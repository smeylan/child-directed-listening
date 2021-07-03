
# Code to replace run_models_across_time.
# This is for GPU/tmuxable scripts.

from utils import load_models, load_splits
import transfomers_bert_completions as transformers_bert_completions

def assemble_across_time_results():
    
    """
    Assemble the "score store" across models that was present in the original function
        and is used for visualizations.
    Outer loop is by age.
    Inner loop is by model, for that pool.
    Note that different splits have different samples of data.
    """
    
    # First, load the appropriate arguments to call.
    
    model_args = []
    
    # For now, just append whichever ages are available per model.
    # Need to be careful when doing visualizations in yyy later.
    
    
    for model_args in config.model_args_set:
        for use_tags in [True, False]:
            for context in config.context_list:
                this_split, this_dataset_name = model_args

                this_sample = load_splits.load_sample_successes('models_across_time', this_split, this_dataset_name) # What if you don't know the n?

                this_beta_folder = load_beta_folder(this_split, this_dataset_name, use_tags, context)

                # Need to retrieve the ages from the sample -- how?

                this_data_path = join(this_beta_folder, 'run_models_across_time_{age}.csv')
                data_df = pd.read_csv(this_data_path)

    
def successes_across_time_per_model(age, utts, model, all_tokens_phono):
    """
    model = a dict of a model like that in the yyy analysis 
    vocab is only invoked for unigram, which correspond to original yyy analysis.
    
    Unlike original code assume that utts = the sample of utts_with_ages, not the whole dataframe
    """
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_cmu_dict_info()
    
    print('Running model '+model['title']+'...')
    
    selected_success_utts = utts.loc[(utts.set == 'success') 
            & (utts_with_ages.year == age)]
    
    selected_yyy_utts = utts.loc[(utts.set == 'failure') 
            & (utts_with_ages.year == age)]
    
    
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

    if model['type'] == 'BERT':
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, 
            edit_distances_for_age_interval, initial_vocab, beta_value = optimal_beta)
    elif model['type'] == 'unigram':
        # special unigram hack
        posteriors_for_age_interval = transformers_bert_completions.get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, initial_vocab, score_store[-1].bert_token_id, beta_value = optimal_beta)


    posteriors_for_age_interval['scores']['model'] = model['title']
    posteriors_for_age_interval['scores']['age'] = age
    
    return copy.deepcopy(posteriors_for_age_interval['scores'])
