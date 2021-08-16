
import os
from os.path import join, exists
from utils import load_models

import config


# Arguments that control model versioning

version_name = 'lr_search' # {'lr_search' for hyperparameter search}
is_search = version_name == 'lr_search'

lr_search_params = [1e-3, 7.5e-4, 5e-4, 1e-4, 5e-5] 

exp_dir = join(join(config.root_dir, 'experiments'), version_name)
model_dir = join(exp_dir, 'models')

# Arguments that control taking a subset of the dataset

#cut_ratio = 0.25 # Take 1/4 of all of the non-child text files for iterating on training to convergence.
cut_ratio = 46000 # For the hyperparam searhc

use_full_text = False

finetune_cut_dir_name = f'finetune_cut_{cut_ratio}'
finetune_run_dir_name = finetune_cut_dir_name if not use_full_text else config.finetune_dir_name
finetune_run_path = join(config.root_dir, finetune_run_dir_name)

# Wandb arguments -- not currently used.

wandb_user = 'w-nicole'
project_name = "child-directed-listening" 

#########################
#### CHILD ARGUMENTS ####
#########################


child_lr = 1e-4
child_interval = 10
child_epochs = 10

child_args = {
    
    'model_name_or_path' : load_models.get_model_path('all', 'all', True),
    
    'num_train_epochs' : child_epochs,
    'learning_rate' : child_lr, # Unsure, need to check Alex convergence etc.

    'eval_steps' : child_interval,
    'logging_steps' : child_interval,
    'save_steps' : child_interval,
    
}

#########################
## NON-CHILD ARGUMENTS ##
#########################

non_child_lr = 0.00075 # Searched manually, 1000 examples, on no tags with the max_train_samples truncation
non_child_interval = 500
non_child_epochs = 15 # Until convergence, if possible.

non_child_args = {
    
    'model_name_or_path' : 'bert-base-uncased',
    
    'num_train_epochs' : non_child_epochs,
    'learning_rate' : non_child_lr,
    
    'eval_steps' : non_child_interval,
    'logging_steps' : non_child_interval,
    'save_steps' : non_child_interval,
    
    
}


### Base arguments

batch_size = 8 # Maximal for linebyline = False, 9 GB GPU.
interval_steps = 500 

base_args = {
    
    'model_name_or_path' : 'bert-base-uncased',
    
    # Boolean arguments: basically pass in the argument --do_train, which signifies True
    'do_train' : '', 
    'do_eval': '',
    'load_best_model_at_end' : '',
    'overwrite_output_dir' : '',
    
    'metric_for_best_model' : 'eval_loss',
    
    'evaluation_strategy' : 'steps',
    'save_strategy' : 'steps',
    'logging_strategy' : 'steps',
    'save_total_limit' : 1,
    
    # Always overwrite by default. Note child arguments load from a model path, not a trainer checkpoint.
    
    'per_device_train_batch_size' : batch_size,
    'per_device_eval_batch_size'  : batch_size,
    
    'weight_decay': 1e-7, 
    
}

# Update the specific args with the base args

non_child_args.update(base_args)
child_args.update(base_args)

