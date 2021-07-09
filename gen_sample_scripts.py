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
    
    task_names = ['beta_search', 'models_across_time']
    task_files = ['run_beta_search.py', 'run_models_across_time.py']
    
    
    for task_name, task_file in zip(task_names, task_files):
        
        sh_script_loc = join(config.root_dir, f'scripts_{task_name}') # Note you don't want to submit training with beta search all together at the same time by accident.

        if not exists(sh_script_loc):
            os.makedirs(sh_script_loc)

        commands = scripts.gen_command_header(time_alloc_hrs = 2)

        for arg_set in model_args:

            split, dataset, use_tags, context_width, model_type = arg_set

            model_id = load_models.get_model_id(
                split, dataset, use_tags, context_width, model_type
            ).replace('/', '>')
            command = f"python3 {task_file} --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}" # This may have to be "python3" on openmind? 

            command = scripts.gen_singularity_header() + command

            with open(join(sh_script_loc, f'{task_name}_{model_id}.sh'), 'w') as f:
                f.writelines(commands + [command])
    
    
    
    

    