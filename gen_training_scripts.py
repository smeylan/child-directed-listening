# Used to deploy training and automatically request consistent text data.

import os
from os.path import join, exists

from utils import scripts
import config

from datetime import datetime

def models_get_split_folder(split_type, dataset_type, with_tags, datetime_str, base_dir = config.om_root_dir):
    
    tags_str = 'with_tags' if with_tags else 'no_tags' # For naming the model folder
    
    base_dir = join(base_dir, f'experiments/models/{datetime_str}') 
    return join(base_dir, join(join(split_type, dataset_type), tags_str))

def get_training_shell_script(split_name, dataset_name, with_tags, om2_user = 'wongn'):
    """
    Make sure you copy the latest, proper data folder to OM2.
    """
    
    tags_data_str  = '' if with_tags else '_no_tags' # For loading the proper data
    datetime_gen = str(datetime.today()).replace(' ', '_')
    
    this_model_dir = models_get_split_folder(split_name, dataset_name, with_tags, datetime_gen)
    
    this_data_dir = join(config.om_root_dir, join(config.finetune_dir, join(split_name, dataset_name)))
    
    if not exists(this_model_dir) and config.root_dir == config.om_root_dir: # You are on OM
        os.makedirs(this_model_dir)
        
    # For the command text
    # 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
    # and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
    # including the bash line at the top

    commands = scripts.gen_command_header(mem_alloc_gb = 9, time_alloc_hrs = 7 if split_name != 'child' else 3) 

    # 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id
    # Got the variable guidance for what variable name to use for job id
    commands.append("mkdir ~/.cache/$SLURM_JOB_ID\n")
    # end usage of variable
    commands.append("# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID\n")
    commands.append(f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/trans-pytorch-gpu \
    python3 run_mlm.py \
            --model_name_or_path bert-base-uncased \
            --do_train \
            --do_eval \
            --output_dir {this_model_dir}\
            --train_file {this_data_dir}/train{tags_data_str}.txt \
            --validation_file {this_data_dir}/val{tags_data_str}.txt \
            --cache_dir ~/.cache/$SLURM_JOB_ID \
            --overwrite_output_dir")
    
    # end 7/13/21
    # end taken command code 6/24/21

    commands.append("\n# end taken command code 6/24/21 and slurm id reference 7/13/21")

    return commands


def get_script_name(split, dataset, is_tags):
    
    this_tags_str = 'with_tags' if is_tags else 'no_tags'
    return f'run_model_{split}_{dataset}_{this_tags_str}.sh'
    
def write_training_shell_script(split, dataset, is_tags, om2_user = 'wongn'): 
    
    script_dir = join(config.root_dir, 'scripts_train')
    
    if not exists(script_dir):
        os.makedirs(script_dir)
    
    script_name = get_script_name(split, dataset, is_tags)
    
    with open(join(script_dir, script_name), 'w') as f:
        f.writelines(get_training_shell_script(split, dataset, is_tags, om2_user = om2_user))
    
    
if __name__ == '__main__':
    
    all_splits = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    for split_args in all_splits:
        for has_tags in [True, False]:
            t_split, t_dataset = split_args
            write_training_shell_script(t_split, t_dataset, has_tags)
            
