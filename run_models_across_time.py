
from utils import load_models
from sample_models_across_time import successes_across_time_per_model


def load_sample_model_across_time_args(model):
    """
    How to load correct arguments for a given split?
    """
    pass

if __name__ == '__main__':
    
    all_models = load_models.get_model_dict()
    # Can you run this subprocess-style? What is best?
    
    
    utts_with_ages, 
    all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab
    
    # Load the appropriate 
    ages = np.unique(utts_with_ages.year)
    for age in ages:
        
        for this_model_dict in models:
            successes_across_time_per_model(age, utts, )
    pass