
import copy

from utils import load_splits, load_models, transformers_bert_completions
from utils_model_sampling import beta_utils
from utils_child import child_models

import config

import os
from os.path import join, exists

def get_cross_path(data_child_name, prior_child_name, beta):
    
    this_folder = join(config.exp_dir, 'child_cross')
    
    if not exists(this_path):
        os.makedirs(this_folder)
    
    this_path = join(this_folder, f'data_{data_child_name}_prior_{prior_child_name}_beta_{beta}.pkl')
    return this_path
    
def score_cross_prior(data_child, prior_child):
    
    """
    Calculate one child's posterior distr. on their data
        with the prior of another child.
    """
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    _, is_tags = child_models.get_best_child_base_model_path()
    
    print('Re-enable optimal beta for child once these values become available!')
    
    # optim_beta = beta_utils.get_optimal_beta_value('child', prior_child, is_tags, 0, 'childes')
    optim_beta = 3.2
    
    # Load the evaluation successes and yyy for a given child.
    eval_data = load_splits.load_pvd_data('child', data_child, config.eval_phase)
    
    # Load the prior
    model = child_models.get_child_model_dict(prior_child)
    
    # Use id, not utterance id, because this is Providence second query data.
    cross_priors = transformers_bert_completions.compare_successes_failures(eval_data['phono'], eval_data['success_utts'].utterance_id, eval_data['yyy_utts'].utterance_id, **model['kwargs'])
    
    
    # Calculate distances -- depending on how implementation is done hopefully can abstract this out.
    
    dists = None
    
    if config.dist_type == 'levdist':
        dists = transformers_bert_completions.get_edit_distance_matrix(eval_data['phono'], 
            cross_priors, initial_vocab, cmu_in_initial_vocab)    
    else:
        assert False, "Invalid dist specified in config file. Choose from: {levdist}"
    
    posteriors_for_age_interval = transformers_bert_completions.get_posteriors(cross_priors, 
                    dists, initial_vocab, None, optim_beta)
    
    posteriors_for_age_interval['scores']['beta_value'] = optim_beta
    posteriors_for_age_interval['scores']['model'] = model['title']
        
    scores = copy.deepcopy(posteriors['scores'])
    
    return scores, optim_beta
    