# Generate the scripts for a beta search.

# What is submit.sh here?
# May be worth writing a submit.sh to automatically submit all of your scripts in a given folder.

# Unless GPU running is very, very fast -- somewhat doubtful?

import config
import argparse
from utils import parsers, load_models, scripts

import os
from os.path import join, exists



def gen_commands(task_file, split, dataset, use_tags, context_width, model_type):
    
    commands = scripts.gen_command_header(mem_alloc_gb = 22, time_alloc_hrs = 5)
    # 13 GB approx is required to store a potential CSV (estimated?)
    # Therefore, need probably around 22 GB (regular memory request)

    model_id = load_models.get_model_id(
        split, dataset, use_tags, context_width, model_type
    ).replace('/', '>')
    command = f"python3 {task_file} --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}" # This may have to be "python3" on openmind? 

    command = scripts.gen_singularity_header() + command

    return model_id, commands + [command]


def write_commands(sh_script_loc, task_name, model_id, commands):
    
    with open(join(sh_script_loc, f'{task_name}_{model_id}.sh'), 'w') as f:
        f.writelines(commands)
    return sh_script_loc
            
            
if __name__ == '__main__':
    
    model_args = load_models.gen_all_model_args()
    
    task_names = ['beta_search', 'models_across_time']
    task_files = ['run_beta_search.py', 'run_models_across_time.py']
    
    
    for task_name, task_file in zip(task_names, task_files):
        
        sh_script_loc = join(config.root_dir, f'scripts_{task_name}')

        if not exists(sh_script_loc):
            os.makedirs(sh_script_loc)

        for arg_set in model_args:
            
            model_id, commands = gen_beta_commands(task_file, *arg_set)
            write_commands(sh_script_loc, task_name, model_id, commands)
            
            
            