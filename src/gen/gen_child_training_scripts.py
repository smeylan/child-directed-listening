import os
from os.path import join, exists
import json
import copy
from datetime import datetime
from os.path import join, exists
import sys
import copy

sys.path.append('.')
sys.path.append('src/.')
from src.utils import scripts,  configuration, load_models, paths, training
config = configuration.Config()


if __name__ == '__main__':
    
    task_name = 'child'    
    task_phase = 'train'
    
    child_models = load_models.gen_child_model_args()     
    # only training new models for children. In fitting and eval, I cross child models, adult models, and child test sets 

    child_models_no_context = []
    for model in child_models:        
        model['context_width'] = None #training doesn't manipulate context
        model['task_name'] = task_name
        model['test_split'] = None
        model['test_dataset'] = None # training datast is test dataset
        model['task_phase'] = task_phase
        model['use_tags'] = True
        model['n_samples'] = config.n_across_time                

        child_models_no_context.append(copy.copy(model))
    
    # take the unique models, collapsing across values of context
    unique_child_models = [dict(s) for s in set(frozenset(d.items()) for d in child_models_no_context)]

    for model in child_models_no_context:

        fit_file, fit_commands = training.gen_training_commands(model)

        with open(fit_file, 'w') as f:
            f.writelines(fit_commands)
    
    scripts.gen_submit_script(task_name, task_phase)
