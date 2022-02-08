
import os
from os.path import join, exists
import json
import shutil

from utils import load_models, load_splits

import configuration
config = configuration.Config()



def get_best_child_base_model_path(which_metric = 'perplexity'):
    """
    Between the CHILDES no tags and with tags all/all models, choose the one with lower val perplexity.
    """
    
    assert which_metric in {'perplexity', 'eval_loss'}
    
    with_tags_path = load_models.get_model_path('all', 'all', False)
    no_tags_path = load_models.get_model_path('all', 'all', True)
    
    which_results = {} 
    for tags_str, tags_path in zip(['with', 'no'], [with_tags_path, no_tags_path]):
        with open(join(tags_path, 'all_results.json'), 'r') as f:
            which_results[tags_str] = json.load(f)[which_metric]
    
    if which_results['with'] >= which_results['no']:
        return with_tags_path, True
    else:
        return no_tags_path, False

def get_child_names():
    """
    Get all Providence children.
    """
    
    all_phono = load_splits.load_phono()
    return sorted(list(set(all_phono.target_child_name)))
   
    
def get_child_model_dict(name):
    
    _, is_tags = get_best_child_base_model_path()
    
    model_args = ('child', name, is_tags, config.child_context_width, 'childes')
     
    model_id = load_models.get_model_id(*model_args)
    
    model_path = load_models.get_model_path('child', name, is_tags)
          
    model_dict = {
        'title' : load_models.gen_model_title(*model_args),
        'kwargs' : load_models.get_model_from_path(model_path, is_tags),
        'type' : 'BERT', 
    }
    
    model_dict['kwargs'].update({'context_width_in_utts' : 0, 'use_speaker_labels' : is_tags})
   
    return model_dict
    