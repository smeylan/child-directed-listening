# Finding a problematic sample on OM -- debug purposes only.

from utils import load_splits, load_models
from utils_model_sampling import sample_models_across_time 

if __name__ == '__main__':

    this_id = load_models.get_model_id('all', 'all', True, 20, 'childes')
    this_model_dict = load_models.get_model_dict()[this_id]

    # Load the successes
    this_age = 4.0
    this_pool_ids = load_splits.load_sample_successes('models_across_time', 'all', 'all', age = this_age)

    # Need to index into this pool? How to do so? Make a function to do this?
    
    eval_data = load_splits.load_eval_data_all('all', 'all')
    tokens = eval_data['phono']
    all_pool = pd.concat([eval_data['success_utts'], eval_data['yyy_utts']])
    
    this_pool = all_pool[all_pool.utterance_id.isin(this_pool_ids)]
    
    # Use any beta for now because this is just debugging.
    this_scores = sample_models_across_time.successes_and_failures_across_time_per_model(this_age, this_pool, this_model_dict, tokens, beta_value = 3.2)
    
    # What to do here? Just view the output until it errors out.
    # If you can get utterance id
    
    
