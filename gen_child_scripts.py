    
import os
from os.path import join, exists

import gen_training_scripts, gen_sample_scripts
import config

from utils_child import child_models
from utils import split_gen, scripts

def gen_child_commands(name, base_model_path, is_tags):
    
    your_model_path = split_gen.get_split_folder('child', name, config.model_dir)
    
    # Get the directory of this model so rsync works correctly
    this_model_dir = '/'.join(gen_training_scripts.get_versioning('child', name, is_tags).split('/')[:-1])
    
    copy_commands = [
        f"\nrm -r {this_model_dir}\n"
        # 7/15/21: rsync advice and command
        # https://askubuntu.com/questions/86822/how-can-i-copy-the-contents-of-a-folder-to-another-folder-in-a-different-directo
        f"\nrsync -a --progress {base_model_path} {this_model_dir}",
        # end rsync
    ]

    # will automatically switch time/mem usage to beta search type.
    train_commands = gen_training_scripts.get_isolated_training_commands('child', name, is_tags)

    # Put the bin bash header and the citations at the front        
    commands = train_commands[:2] + copy_commands + train_commands[2:-1]
    
    # Drop the "end cite" command at the end, get the final training command
    # Need to edit the last command to have the beta search attached.
    
    sing_header = scripts.gen_singularity_header()
    
    commands[-1] = commands[-1] + f"; {sing_header} {gen_sample_scripts.get_one_python_command('run_beta_search.py', 'child', name, is_tags, 0, 'childes')[1]}"
    
    filename = scripts.get_script_name('child', name, is_tags)
    
    return filename, commands

    
if __name__ == '__main__':
    
    child_names = child_models.get_child_names()
    
    sh_train_loc = join(config.root_dir, 'scripts_child_train_beta')
    
    if not exists(sh_train_loc):
        os.makedirs(sh_train_loc)

    _, is_tags = child_models.get_best_child_base_model_path()
    base_model_path = gen_training_scripts.get_versioning('all', 'all', is_tags)
    
    # Need to convert to OM
    
    for child in child_names:
    
        # Generate appropriate scripts for model_training
        
        train_file, train_commands = gen_child_commands(child, base_model_path, is_tags)
        
        with open(join(sh_train_loc, train_file), 'w') as f:
            f.writelines(train_commands)
    