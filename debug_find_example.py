# Finding a problematic sample on OM -- debug purposes only.

from utils import load_splits, load_models
import sample_models_across_time 

if __name__ == '__main__':

    this_id = load_models.get_model_id('all', 'all', True, 20, 'childes')
    this_model_dict = load_models.get_model_dict()[this_id]

    # Load the successes
    this_age = 4.0
    this_pool = load_splits.load_sample_successes('models_across_time', 'all', 'all', age = this_age)

    tokens = load_splits.load_eval_data_all('all', 'all')['phono']
    
    # Use any beta for now because this is just debugging.
    this_scores = sample_models_across_time.successes_and_failures_across_time_per_model(this_age, this_pool, this_model_dict, tokens, beta_value = 3.2)
    
    # What to do here? Just view the output until it errors out.
    # If you can get utterance id
    
    
