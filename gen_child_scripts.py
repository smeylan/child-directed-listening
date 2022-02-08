    
import os
from os.path import join, exists

import gen_training_scripts, gen_sample_scripts
import configuration
config = configuration.Config()

from utils_child import child_models
from utils import split_gen, scripts

    
def gen_child_commands(name, is_tags):
    
    your_model_path = split_gen.get_split_folder('child', name, config.model_dir)
    
    # ---------- begin new code
    
    # Generate the appropriate header and the slurm folder
    
    slurm_folder = scripts.get_slurm_folder('child', name, 'child_train')
    
    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = gen_training_scripts.get_training_alloc('child')
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        slurm_folder = slurm_folder,
        slurm_name = f'training_beta_tags={is_tags}', 
        two_gpus = False)
    
    this_model_dir = '/'.join(gen_training_scripts.models_get_split_folder('child', name, is_tags).split('/')[:-1])
    
    # Construct the python/training-related commands
    
    run_commands = gen_training_scripts.get_non_header_commands('child', name, is_tags)[:-1] 
    
    ## Edit the last command to append the beta search.
    sing_header = scripts.gen_singularity_header()
    
    run_commands[-1] = run_commands[-1] + f"; {sing_header} {gen_sample_scripts.get_one_python_command('run_beta_search.py', 'child', name, is_tags, 0, 'childes')[1]}\n"

    # Put the copy commands between the header and the actual python runs.
    
    commands = header_commands + [f"rm -r {this_model_dir}\n"] + run_commands
    
    filename = scripts.get_script_name('child', name, is_tags)
    
    return filename, commands

    
if __name__ == '__main__':
    
    _, is_tags = child_models.get_best_child_base_model_path()
    child_names = child_models.get_child_names()
    
    task_name = 'child_train'
    
    sh_train_loc = f'scripts_{task_name}'
    
    child_arg_list = [('child', name) for name in child_names]
    scripts.gen_submit_script(task_name, child_arg_list, 'child_train')
    
    if not exists(sh_train_loc):
        os.makedirs(sh_train_loc)

    for child in child_names:
    
        # Generate appropriate scripts for model_training
        
        train_file, train_commands = gen_child_commands(child, is_tags)
        
        with open(join(sh_train_loc, train_file), 'w') as f:
            f.writelines(train_commands)
   
    