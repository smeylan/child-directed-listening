# Used to deploy training and automatically request consistent text data.

import os
from os.path import join, exists
import json

from utils import scripts, split_gen
from utils_child import child_models

from utils_model_analysis import hyperparams

import config
import config_train

from datetime import datetime

import gen_sample_scripts

def models_get_split_folder(split_type, dataset_type, with_tags, base_dir = config.om_root_dir):
    
    tags_str = 'with_tags' if with_tags else 'no_tags' # For naming the model folder
    
    base_dir = join(base_dir, f'experiments/{config_train.version_name}/models') 
    return join(base_dir, join(join(split_type, dataset_type), tags_str))


def get_training_alloc(split_name):
    
    if split_name == 'child':
        # Need to run beta simultaneously
        time, mem = gen_sample_scripts.time_and_mem_alloc()
        mem_alloc_gb = mem
        time_alloc_hrs = time
    else:
        mem_alloc_gb = 9 
        time_alloc_hrs = 6 if not config.dev_mode else (0, 20, 0)
    
    return time_alloc_hrs, mem_alloc_gb
    
    
def get_training_header_commands(split_name, dataset_name, with_tags, om2_user = config.om_user, lr = None):
    
    time_alloc_hrs, mem_alloc_gb = get_training_alloc(split_name)
    
    model_dir = models_get_split_folder(split_name, dataset_name, with_tags)
    
    print('Use of cvt root dir will not be compatible with eventual updating datetime in gen_training_scripts')
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
                                          slurm_folder = scripts.get_slurm_folder(split_name, dataset_name, task = 'non_child_train'),
                                          slurm_name = f'training_tags={with_tags}{f"_{lr}" if lr is not None else ""}', 
                                          two_gpus = (dataset_name in {'young', 'all'}))
    return header_commands
    
    
def get_isolated_training_commands(split_name, dataset_name, with_tags, om2_user = config.om_user):
    
    header_commands = get_training_header_commands(split_name, dataset_name, with_tags, om2_user)
    non_header_commands = get_non_header_commands(split_name, dataset_name, with_tags, om2_user)
    
    commands = header_commands + non_header_commands
    
    return commands


def get_run_mlm_command(split_name, dataset_name, this_data_dir, this_model_dir, tags_data_str, om2_user):
    
    this_args_dict = config_train.child_args if split_name == 'child' else config_train.non_child_args
    
    if split_name == 'child':
        _, is_tags = child_models.get_best_child_base_model_path()
        base_model = models_get_split_folder('all', 'all', is_tags)
    else:
        base_model = 'bert-base-uncased'
        
    this_args_dict['model_name_or_path'] = base_model
    
    this_args_list = sorted(list(this_args_dict.keys())) # readability
    
    data_args = [
            f"--train_file {this_data_dir}/train{tags_data_str}.txt",
            f"--validation_file {this_data_dir}/val{tags_data_str}.txt", 
            f"--cache_dir ~/.cache/$SLURM_JOB_ID",
            f"--output_dir {this_model_dir}",
        ]
    
    trainer_args = [
        f"--{key} {this_args_dict[key]}"
        for key in this_args_list
    ]
    
    if config.dev_mode:
        trainer_args += [
            f"--max_train_samples 10",
            f"--max_eval_samples 10",
        ]

    main_command = f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/trans-pytorch-gpu"
    this_python_command = f' python3 run_mlm.py {" ".join(data_args + trainer_args)}'

    return f"{main_command}{this_python_command}"
    

def get_non_header_commands(split_name, dataset_name, with_tags, om2_user = config.om_user):
    
    tags_data_str  = '' if with_tags else '_no_tags' # For loading the proper data
    
    model_dir = models_get_split_folder(split_name, dataset_name, with_tags)
    
    data_dir = join(config.om_root_dir, join(config.finetune_dir, join(split_name, dataset_name)))
        
    commands = []
   
    # For the command text
    # 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
    # and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
    # including the bash line at the top

    # 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id
    # Got the variable guidance for what variable name to use for job id
    commands.append("mkdir ~/.cache/$SLURM_JOB_ID\n")
    # end usage of variable
    commands.append("# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID\n")
    
    main_command_all = get_run_mlm_command(split_name, dataset_name, data_dir, model_dir, tags_data_str, om2_user)
    
    commands.append(main_command_all)
    
    # end 7/13/21
    # end taken command code 6/24/21

    commands.append("\n# end taken command code 6/24/21 and slurm id reference 7/13/21")
    return commands


if __name__ == '__main__':
    
    label = 'non_child_train'
    
    all_splits = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    for split_args in all_splits:
        for has_tags in [True, False]:
            t_split, t_dataset = split_args
            tags_str = 'with_tags' if has_tags else 'no_tags'
            scripts.write_training_shell_script(t_split, t_dataset, has_tags, f'scripts_{label}/{tags_str}', get_isolated_training_commands)

    scripts.gen_submit_script(label, config.childes_model_args, label)
    