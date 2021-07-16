

from utils import load_splits
from utils_model_sampling import beta_utils

import config

def find_best_model():
    
    # Need to compare the with_tags and no_tags models, choose the one with lower perplexity,
    # then copy it for finetuning automatically to the right child folder.
    
    # Then, need to prep scripts for child finetuning
    
    # Then, need to score one child's productions via another child's -- but how?
    
    # for the priors/likelihoods:
    # for one child
    # do this: compare_successes_failures -> select on all utterance ids
    # then this: get_edit_distance_matrix
    
    
    # then, you should somehow score the two separately -- but it's not obvious what the partition in the data is.
    
    
    
    
    
def score_cross_prior(data_child, prior_child):
    
    """
    Calculate one child's posterior distr. on their data
        with the prior of another child.
    """
    
    
    # For the priors/etc you actually need to?
    # Note for the run across beta etc. you need to...
    # You need to load the entirety of the child data to perform the optimization
    # So it may be different than the current code.
    
    # Specifically for load_sample_successes, you will just have to recopy all the data to that location
    # (optimize on the entire split)
    
    # Load the proper beta value for the given child.
    
    # How to load the right model dictionary here? 
    
    optim_beta = beta_utils.get_optimal_beta_value('child', prior_child, True, 0, 'childes')
    print('Update this to use best model, not just speaker tags/no context')
    
    # You should load the better model here -> use the config files to decide if you can.
    
    # Load the evaluation successes and yyy for a given child.
    eval_data = load_splits.load_child_eval_data(data_child)
    # prior_model = load_models.
    
    # Pass the evaluation set to scoring 
    # What to load for all_tokens_phono?
    # You need to separate yyy, all, and success for the eval set itself.
    # How to avoid confusing naming conventions?
    
    # Use id, not utterance id, because this is Providence second query data.
    priors = compare_successes_failures(eval_data['phono'], eval_data['success_utts'].id, eval_data['yyy_utts'].id, **model['kwargs'])
    
    # This will need to be 
    
    
    # Load the success and yyy for a given child.
    
def 
    