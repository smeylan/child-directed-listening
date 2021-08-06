    
import os
from os.path import join, exists

import gen_training_scripts, gen_sample_scripts
import config

from utils_child import child_models
from utils import split_gen, scripts

def gen_child_commands(name, base_model_path, is_tags):
    
    your_model_path = scripts.cvt_root_dir('child', name, config.model_dir)
    

    
    # ---------- begin new code
    
    # Generate the appropriate header and the slurm folder
    
    time_alloc_hrs, mem_alloc_gb = gen_training_scripts.get_training_alloc('child')
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
                                          slurm_folder = scripts.cvt_root_dir('child', name, config.scores_dir),
                                          slurm_name = f'training_beta_tags={is_tags}', 
                                          two_gpus = False)
    
    # Reset and rsync the right model configuration over
    
    ## Get the directory of this model so rsync works correctly
    this_model_dir = '/'.join(gen_training_scripts.get_versioning('child', name, is_tags).split('/')[:-1])

    copy_commands = [
        f"\nrm -r {this_model_dir}\n"
        # 7/15/21: rsync advice and command
        # https://askubuntu.com/questions/86822/how-can-i-copy-the-contents-of-a-folder-to-another-folder-in-a-different-directo
        f"\nrsync -a --progress {base_model_path} {this_model_dir}\n",
        # end rsync
    ]
    
    commands = header_commands + copy_commands
    
    # Construct the python/training-related commands
    
    run_commands = gen_training_scripts.get_non_header_commands('child', name, is_tags)[:-1] 
    
    ## Edit the last command to append the beta search.
    sing_header = scripts.gen_singularity_header()
    
    run_commands[-1] = run_commands[-1] + f"; {sing_header} {gen_sample_scripts.get_one_python_command('run_beta_search.py', 'child', name, is_tags, 0, 'childes')[1]}\n"

    # Put the copy commands between the header and the actual python runs.
    
    commands = header_commands + copy_commands + run_commands
    
    filename = scripts.get_script_name('child', name, is_tags)
    
    return filename, commands

    
if __name__ == '__main__':
    
    child_names = child_models.get_child_names()
    
    sh_train_loc = join(config.root_dir, 'scripts_child_train_beta')
    
    if not exists(sh_train_loc):
        os.makedirs(sh_train_loc)

    _, is_tags = child_models.get_best_child_base_model_path()
    base_model_path = gen_training_scripts.get_versioning('all', 'all', is_tags)
    
    for child in child_names:
    
        # Generate appropriate scripts for model_training
        
        train_file, train_commands = gen_child_commands(child, base_model_path, is_tags)
        
        with open(join(sh_train_loc, train_file), 'w') as f:
            f.writelines(train_commands)
    