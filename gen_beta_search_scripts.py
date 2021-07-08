# Generate the scripts for a beta search.

# What is submit.sh here?
# May be worth writing a submit.sh to automatically submit all of your scripts in a given folder.

# Unless GPU running is very, very fast -- somewhat doubtful?

import config
import argparse
from utils import parsers, load_models, scripts

import os
from os.path import join, exists

if __name__ == '__main__':
    
    model_args = load_models.gen_all_model_args()
    
    print(model_args)
    
    sh_script_loc = join(config.root_dir, 'scripts_beta_search') # Note you don't want to submit training with beta search all together at the same time by accident.
    if not exists(sh_script_loc):
        os.makedirs(sh_script_loc)
        
    commands = scripts.gen_command_header(time_alloc_hrs = 2)
    
    for arg_set in model_args:
        
        split, dataset, use_tags, context_width, model_type = arg_set
            
        model_id = load_models.get_model_id(
            split, dataset, use_tags, context_width, model_type
        ).replace('/', '>')
        command = f"python3 run_beta_search.py --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}" # This may have to be "python3" on openmind? 
        
        command = scripts.gen_singularity_header() + command
        
        with open(join(sh_script_loc, f'beta_search_{model_id}.sh'), 'w') as f:
            f.writelines(commands + [command])
            
    # Generate the unigram scripts separately
    
    
    
    
    

    