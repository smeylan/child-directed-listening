import os
import sys
from os.path import join, exists
sys.path.append('.')
sys.path.append('src/.')


sys.path.append('.')
sys.path.append('src/.')
from src.gen import gen_training_scripts, gen_sample_scripts
from src.utils import split_gen, scripts, configuration, child_models
config = configuration.Config()

    
def gen_child_training_commands(model_child, data_child, is_tags):
    
    your_model_path = split_gen.get_split_folder('child', model_child, config.model_dir)
    
    # ---------- begin new code
    
    # Generate the appropriate header and the slurm folder
    
    slurm_folder = scripts.get_slurm_folder('child', model_child, 'child_train')
    
    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = gen_training_scripts.get_training_alloc('child')
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        slurm_folder = slurm_folder,
        slurm_name = f'training_beta_tags={is_tags}', 
        two_gpus = False)
    commands = header_commands


    this_model_dir = '/'.join(gen_training_scripts.models_get_split_folder('child', model_child, is_tags).split('/')[:-1])
    

    sing_header = scripts.gen_singularity_header()
    
    commands += [f"rm -r {this_model_dir}\n"]  # clear the directory to train new stuff     
        
    
    # Construct the python/training-related commands
    run_commands = gen_training_scripts.get_non_header_commands('child', model_child, is_tags)[:-1]
        
    # Put the copy commands between the header and the actual python runs.
    commands += run_commands
    
    filename = scripts.get_script_name('child', model_child, is_tags, data_child)

    return filename, commands

    
if __name__ == '__main__':
    
    _, is_tags = child_models.get_best_child_base_model_path()
    child_names = child_models.get_child_names()
    
    task_name = 'child_train'
    
    sh_train_loc = f'output/SLURM/scripts_{task_name}'
    
    child_arg_list = [('child', name) for name in child_names]
    scripts.gen_submit_script(task_name, child_arg_list, 'child_train')
    
    if not exists(sh_train_loc):
        os.makedirs(sh_train_loc)

    for model_child in child_names:
        data_child =  model_child
        # Generate appropriate scripts for model_training
    
        train_file, train_commands = gen_child_training_commands(model_child, data_child, is_tags)
    
        with open(join(sh_train_loc, train_file), 'w') as f:
            f.writelines(train_commands)

