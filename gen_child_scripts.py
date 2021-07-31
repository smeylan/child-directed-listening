    
import os
from os.path import join, exists

import gen_training_scripts, gen_sample_scripts
import config

from utils_child import child_models
from utils import split_gen

def gen_child_train_commands(name, base_model_path, is_tags):
    
    your_model_path = split_gen.get_split_folder('child', name, config.model_dir)
    
    copy_commands = [
        # 7/15/21: rsync advice and command
        # https://askubuntu.com/questions/86822/how-can-i-copy-the-contents-of-a-folder-to-another-folder-in-a-different-directo
        f'\nrsync -a {base_model_path} {join(config.model_dir, "child")}',
        # end rsync
        f'\nmv {base_model_path} {your_model_path}',
    ]

    train_commands = gen_training_scripts.get_training_shell_script('child', name, is_tags)

    # Put the bin bash header and the citations at the front        
    commands = train_commands[:2] + copy_commands + train_commands[2:]
    filename = gen_training_scripts.get_script_name('child', name, is_tags)
    
    return filename, commands


#def gen_child_cross_scripts():
    
    
if __name__ == '__main__':
    
    child_names = child_models.get_child_names()
    
    sh_train_loc = join(config.root_dir, 'scripts_child_train')
    sh_beta_loc = join(config.root_dir, 'scripts_child_beta')
    
    for path in [sh_train_loc, sh_beta_loc]:
        if not exists(path):
            os.makedirs(path)
    
    base_model_path, is_tags = child_models.get_best_child_base_model_path()
    
    for child in child_names:
    
        # Generate appropriate scripts for model_training
        
        train_file, train_commands = gen_child_train_commands(child, base_model_path, is_tags)
        gen_sample_scripts()
        
        with open(join(sh_train_loc, train_file), 'w') as f:
            f.writelines(train_commands)
    
        # Generate scripts for beta searching.
        model_id, commands = gen_sample_scripts.gen_commands('run_beta_search.py', 22, 'child', child, is_tags, config.child_context_width, 'childes')
        gen_sample_scripts.write_commands(sh_beta_loc, 'beta_search', model_id, commands)
        
        
    # Need to generate scripts of cross-child evaluations later.