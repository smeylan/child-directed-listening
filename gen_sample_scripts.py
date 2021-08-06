# Generate the scripts for a beta search.

import config
import argparse
from utils import parsers, load_models, scripts

import os
from os.path import join, exists

def get_one_python_command(task_file, split, dataset, use_tags, context_width, model_type):
    
    model_id = load_models.get_model_id(
        split, dataset, use_tags, context_width, model_type
    ).replace('/', '>')
    
    command = f"python3 {task_file} --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}"

    return model_id, command


def time_and_mem_alloc():
    
    is_subsample = (config.n_subsample <= 500) # Always use n_subsample, just depends if 500 or 1000
    
    # Unsure if it will actually run in 4 hours, but based on old BERT statistics.
    
    this_time_alloc = (0, 45, 0) if is_subsample else (2, 30, 0)
    this_mem_amount = 13 if is_subsample else 35
    
    return this_time_alloc, this_mem_amount
    

def write_commands(sh_script_loc, task_name, model_id, commands):
    
    with open(join(sh_script_loc, f'{task_name}_{model_id}.sh'), 'w') as f:
        f.writelines(commands)
    return sh_script_loc
            
            
if __name__ == '__main__':
    

    task_names = ['beta_search', 'models_across_time']
    task_files = ['run_beta_search.py', 'run_models_across_time.py']
     
    
    sh_script_loc_base = join(config.root_dir, 'scripts_beta_time')

    partitions = {
        'finetune' : load_models.gen_finetune_model_args,
        'shelf' : load_models.gen_shelf_model_args,
    }
    
    print('Generating sample scripts again!')
    
    for partition_name, model_args in partitions.items():
        
        sh_script_loc = join(sh_script_loc_base, partition_name) 
            
        if not exists(sh_script_loc):
            os.makedirs(sh_script_loc)


        # TODO: Adapt this to have variable running times -- especially for data unigram and BERT.
        # "subsampling amount if else non-subsampling amount"
        
        this_time_alloc, this_mem_amount = time_and_mem_alloc()
        
        for arg_set in model_args():
            
            split, dataset, _, _, _ = arg_set
            slurm_folder = scripts.cvt_root_dir(split, dataset, config.scores_dir) 
            
            py_commands = {}

            header = scripts.gen_command_header(mem_alloc_gb = this_mem_amount,
                                                time_alloc_hrs = this_time_alloc,
                                                slurm_folder = slurm_folder,
                                                slurm_name = 'beta_time',
                                                two_gpus = False)

            for task_name, task_file in zip(task_names, task_files):
                model_id, py_commands[task_name] = get_one_python_command(task_file, *arg_set)
            
            # 7/31/21: https://unix.stackexchange.com/questions/552695/how-to-run-multiple-scripts-one-after-another-but-only-after-previous-one-got-co
            sing_header = scripts.gen_singularity_header()
            full_py_command = sing_header + f'{py_commands["beta_search"]}; {sing_header} {py_commands["models_across_time"]}'
            # end cite
            
            all_commands = header + [full_py_command]

            write_commands(sh_script_loc, task_name, model_id, all_commands)

            
            
            