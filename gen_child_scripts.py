    
import os
from os.path import join, exists

import gen_training_scripts, gen_sample_scripts
import config


def gen_child_train_commands(name, base_model_path, is_tags):
    
    your_model_path = split_gen.get_split_folder('child', name, config.model_dir)
    
    copy_command = [
        # 7/15/21: rsync advice and command
        # https://askubuntu.com/questions/86822/how-can-i-copy-the-contents-of-a-folder-to-another-folder-in-a-different-directo
        f'\nrsync -a {base_model_path} {join(config.model_dir, 'child')}',
        # end rsync
        f'\nmv {base_model_path} {your_model_path}',
    ]

    train_commands = gen_training_scripts.get_training_shell_script('child', name, is_tags)

    # Put the bin bash header and the citations at the front        
    commands = train_commands[:2] + copy_commands + train_commands[2:]
    filename = gen_training_scripts.get_script_name('child', name, is_tags)
    
    return filename, commands

def gen_child_beta_commands(name, base_model_path, is_tags, beta_loc):
    
    args = ('run_beta_search.py', 'child', name, is_tags, 0, 'childes')
    model_id, commands = gen_sample_scripts.gen_commands(*args)
    
    
    return model_id, commands

        
if __name__ == '__main__':
    
    child_names = child_models.get_child_names()
    
    sh_base_loc = join(config.root_dir, f'scripts_{task_name}')
    sh_train_loc = join(sh_base_loc, 'train')
    sh_beta_loc = join(sh_base_loc, 'beta')
    
    base_model_path, is_tags = child_models.get_best_child_base_model_path()
    
    for child in child_names:
    
        # Generate appropriate scripts for model_training
        
        train_file, train_commands = gen_child_train_commands(name, base_model_path, is_tags)
        
        with open(join(sh_train_loc, train_file), 'r') as f:
            f.writelines(train_commands)
    
        # Generate scripts for beta searching.
        model_id, commands = gen_commands('run_beta_search.py', 'child', child, is_tags, 0, 'childes')
        gen_sample_scripts.write_commands(sh_beta_loc, 'beta_search', model_id)
        
        
    # Need to generate scripts of cross-child evaluations later.