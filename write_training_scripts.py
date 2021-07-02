# Used to deploy training and automatically request consistent text data.

import os
from os.path import join, exists


def scripts_get_split_folder(split_type, dataset_name, base_dir = 'data/new_splits'):
    """
    The same function as in split_gen. This is copied here to prevent having two versions of the same split_gen file.
    """
   
    path = join(base_dir, join(split_type, dataset_name))
    
    if not exists(path):
        os.makedirs(path)
    
    return path

def models_get_split_folder(split_type, dataset_type, with_tags, base_dir = 'data/new_splits'):
    
    tags_str = '_with_tags' if with_tags else '_no_tags' # For naming the model folder
    split_dir = scripts_get_split_folder(split_type, dataset_type, '')
    return join(base_dir, join('models', join(split_dir, tags_str)))

def get_training_shell_script(split_name, dataset_name, with_tags, om2_user = 'wongn'):
    """
    Make sure you copy the latest, proper data folder to OM2.
    """
    
    tags_data_str  = '' if with_tags else '_no_tags' # For loading the proper data
    base_dir = f'/om2/user/{om2_user}/childes_run'
    #base_dir = f'./om2/user/{om2_user}/childes_run' # The version with . is to test script on chompsky
    
    this_split_dir = scripts_get_split_folder(split_name, dataset_name, '')
    this_model_dir = models_get_split_folder(split_name, dataset_name, with_tags, '')
    
    this_data_dir = join(base_dir, join('data/new_splits', this_split_dir))
    
    if not exists(this_model_dir):
        os.makedirs(this_model_dir)
        
    # This needs to be copied from Chompsky to OM2 properly.
    # Should have vocab.csv on all of CHILDES in the data folder, and the new_splits in the folder.
    # Need to clean out the outdated data in the "data" folder later.
    
    # For the command text
    # 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
    # and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
    # including the bash line at the top

    commands = []
    commands.append("#!/bin/bash\n")
    
    # Citation text for every script
    commands.append("\n# For the command text\n# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F\n# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392\n# including the bash line at the top\n")
    
    commands.append("\n#SBATCH -N 1\n")                         
    commands.append("#SBATCH -p cpl\n")
    commands.append("#SBATCH --gres=gpu:1\n")
    commands.append("#SBATCH -t 7:00:00\n")
    commands.append("#SBATCH --mem=9G\n")
    commands.append("#SBATCH --constraint=high-capacity\n")
     
    commands.append("\nmodule load openmind/singularity/3.2.0\n")
    commands.append(f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/trans-pytorch-gpu \
    python3 run_mlm.py \
            --model_name_or_path bert-base-uncased \
            --do_train \
            --do_eval \
            --output_dir {this_model_dir}\
            --train_file {this_data_dir}/train{tags_data_str}.txt \
            --validation_file {this_data_dir}/val{tags_data_str}.txt \
            --overwrite_output_dir")
    
    return commands

def write_training_shell_script(split, dataset, is_tags, om2_user = 'wongn'): 
    
    this_tags_str = '_with_tags' if is_tags else '_no_tags' # For naming the model folder
    
    base_dir = f'/om2/user/{om2_user}/childes_run'
    #base_dir = f'./om2/user/{om2_user}/childes_run' # The version with . is to test script on chompsky
    
    script_dir = join(base_dir, 'scripts')
    
    if not exists(script_dir):
        os.makedirs(script_dir)
    
    script_name = f'run_model_{split}_{dataset}{this_tags_str}.sh'
    
    with open(join(script_dir, script_name), 'w') as f:
        f.writelines(get_training_shell_script(split, dataset, is_tags, om2_user = om2_user))
    
    
if __name__ == '__main__':
    
    # Try testing this shell script generation process on Chompsky first before moving it to OM2 -- but be sure that the commands append will even work... ?
    
    all_splits = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    for split_args in all_splits:
        for has_tags in [True, False]:
            t_split, t_dataset = split_args
            write_training_shell_script(t_split, t_dataset, has_tags)
            
