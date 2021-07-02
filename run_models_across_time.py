
import os
from os.path import join, exists

from utils import load_models, load_splits
from sample_models_across_time import successes_across_time_per_model

import pandas as pd

def load_sample_model_across_time_args(model_name):
    """
    How to load correct arguments for a given split?
    """
    
    utts_filename = None; all_tokens_phono_filename = None
    print('Update the load sample model across time args as new models are added.')
    
    this_utts_save_path = join('eval/new_splits', model_name)
    
    if model_name == 'all/all' or ('meylan/meylan' in model_name) or ('all_old/all_old' in model_name):
        # Note that meylan/meylan and all_old/all_old won't actually fully replicate
        # because original splits are lost for yyy.
        utts_filename = pd.read_csv(join('all/all', ))
        all_tokens_phono_filename = pd.read_csv()
    else if model_name == None:
        utt_filenames
        
    eval_data_dict = load_splits.load_eval_data_all(split_name, dataset_name, base_dir) 
    utts_with_ages = pd.concat([eval_data_dict['success_utts'], eval_data_dict['yyy_utts']])
    this_tokens_phono = eval_data_dict['']
    
    utts = pd.read_csv(utts_filename)
    tokens_phono = pd.read_csv(all_tokens_phono_filename)
    
    if 'no_tags' in model_name:
        
    return utts, tokens_phono

                                    
if __name__ == '__main__':
    
    root_dir = '/home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening'
    
    results_dir = 'intermediate_results/models_across_time'
    
    if not exists(results_dir):
        os.makedirs(results_dir)
        
    all_models = load_models.get_model_dict(root_dir)
    # Can you run this subprocess-style? What is best? For now just run sequentially because GPU.
    
    # How to load the optimal beta value?
    
    # Optimize beta on eval split for age, child analyses.
                                    
    
    
    # Load the appropriate 
    ages = np.unique(utts_with_ages.year)
    
    for age in ages:
        for this_model in models:
            this_scores = successes_across_time_per_model(age, utts_with_ages, all_models[this_model], this_tokens_phono, root_dir, beta_value = optimal_beta)
            # Need to write the scores
            
    pass