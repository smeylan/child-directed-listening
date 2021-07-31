# Generate the scripts for a beta search.

# What is submit.sh here?
# May be worth writing a submit.sh to automatically submit all of your scripts in a given folder.

# Unless GPU running is very, very fast -- somewhat doubtful?

import config
import argparse
from utils import parsers, load_models, scripts

import os
from os.path import join, exists

def get_non_header_commands(task_file, split, dataset, use_tags, context_width, model_type):
    
    model_id = load_models.get_model_id(
        split, dataset, use_tags, context_width, model_type
    ).replace('/', '>')
    
    command = f"python3 {task_file} --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}"

    return model_id, command


def write_commands(sh_script_loc, task_name, model_id, commands):
    
    with open(join(sh_script_loc, f'{task_name}_{model_id}.sh'), 'w') as f:
        f.writelines(commands)
    return sh_script_loc
            
            
if __name__ == '__main__':
    

    task_names = ['beta_search', 'models_across_time']
    task_files = ['run_beta_search.py', 'run_models_across_time.py']
     
    mem_amount = 50
    
    sh_script_loc_base = join(config.root_dir, 'scripts_beta_time')

    partitions = {
        'finetune' : load_models.gen_finetune_model_args,
        'shelf' : load_models.gen_shelf_model_args,
    }
    
    for partition_name, model_args in partitions.items():
        
        sh_script_loc = join(sh_script_loc_base, partition_name)
            
        if not exists(sh_script_loc):
            os.makedirs(sh_script_loc)


        for arg_set in model_args():
            
            py_commands = {}

            header = scripts.gen_command_header(mem_alloc_gb = mem_amount, time_alloc_hrs = 15)

            for task_name, task_file in zip(task_names, task_files):
                model_id, py_commands[task_name] = get_non_header_commands(task_file, *arg_set)
            
            # 7/31/21: https://unix.stackexchange.com/questions/552695/how-to-run-multiple-scripts-one-after-another-but-only-after-previous-one-got-co
            full_py_command = scripts.gen_singularity_header() + f'{py_commands["beta_search"]}; {py_commands["models_across_time"]}'
            # end cite
            
            all_commands = header + [full_py_command]

            write_commands(sh_script_loc, task_name, model_id, all_commands)

            
            
            