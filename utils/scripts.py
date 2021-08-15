
import os
from os.path import join, exists

from utils import split_gen

import config
import config_train

import subprocess


def get_slurm_folder(split, dataset, task):
    
    base_paths = {
        'non_child_train' : config.model_dir, # Non-child train
        'non_child_beta_time' : config.scores_dir, # Non-child beta + time scoring
        
        'child_train' : config.scores_dir, # Child train + beta search
        'child_cross' : config.scores_dir, # Child scoring
    }
    
    assert task in base_paths.keys()
    return cvt_root_dir(split, dataset, base_paths[task])
    
    
def get_slurm_folders_by_args(args, task):
    
    all_paths = []
    
    for this_args in args:
        this_split, this_dataset = this_args
        this_path = get_slurm_folder(this_split, this_dataset, task)
        all_paths.append(this_path)
    
    return sorted(list(set(all_paths)))
    
def gen_submit_script(dir_name, arg_set, task):
    
    text = ['#!/bin/bash']
    
    mkdir_which = get_slurm_folders_by_args(arg_set, task)
    mkdir_commands = [f"mkdir -p {p}" for p in mkdir_which]
    
    text.extend(mkdir_commands)
    
    text.append(f'FILES="./scripts_{dir_name}/*"')
    text.append('for f in $FILES')
    text.append('do')
    text.append('\techo "Processing $f file..."')
    text.append('\tsbatch $f')
    text.append('\tcat "$f"')
    text.append('done')
    
    sh_path = f'submit_{dir_name}.sh'.replace('/', '_') 
    
    give_space = lambda s : f"{s}\n"
    text = list(map(give_space, text))
    
    with open(sh_path, 'w') as f:
        f.writelines(text)
    
    subprocess.call(f'chmod u+x {sh_path}', shell = True)
    
    return text


def cvt_root_dir(split, dataset, base_dir, name = config_train.version_name):
    
    # Convert to the OM root dir while using the Chompsky equivalents to specify folders
    # like scores_dir, etc.
    
    local_path = split_gen.get_split_folder(split, dataset, base_dir)
    
    if config.om_root_dir == config.root_dir:
        return local_path
    
    print('Need to make this more maintainable, cvt root dir converting no versioning to the appropriate name for child work.')
    return local_path.replace(config.root_dir, config.om_root_dir).replace('no_versioning', config_train.version_name)
    
    
def write_training_shell_script(split, dataset, is_tags, dir_name, get_command_func, om2_user = 'wongn'): 
    
    script_dir = join(config.root_dir, dir_name)
    
    if not exists(script_dir):
        os.makedirs(script_dir)
    
    script_name = get_script_name(split, dataset, is_tags)
    
    with open(join(script_dir, script_name), 'w') as f:
        f.writelines(get_command_func(split, dataset, is_tags, om2_user = om2_user))
        

def get_script_name(split, dataset, is_tags):
    
    this_tags_str = 'with_tags' if is_tags else 'no_tags'
    return f'run_model_{split}_{dataset}_{this_tags_str}.sh'

    
# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

    
def gen_singularity_header(om2_user = 'wongn'):
    
    # still part of the taken code above
    return f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/trans-pytorch-gpu " 
    
def format_time(args):
    
    # Skip formatting hours, it should not add a 0 in front of the hours.
    # If a 0 is added it will terminate running early
     
    new_args = tuple([f'0{arg}' if arg < 10 else str(arg) for arg in args[1:]])
    return (args[0],) + new_args
  

def gen_command_header(mem_alloc_gb, time_alloc_hrs, slurm_folder, slurm_name, two_gpus = False):
    
    if isinstance(time_alloc_hrs, int):
        time_alloc_hrs_str = f'{time_alloc_hrs}:00:00'
    if isinstance(time_alloc_hrs, tuple):
        hrs, mins, secs = format_time(time_alloc_hrs)
        time_alloc_hrs_str = f'{hrs}:{mins}:{secs}'
              
    
    slurm_organization_command = f"#SBATCH --output={slurm_folder}/%j.out\n" if slurm_name is None else f"#SBATCH --output={slurm_folder}/%j_{slurm_name}.out\n"
    
    commands = []
    commands.append("#!/bin/bash\n")
    
    # Citation text for every script
    commands.append("\n# For the command text\n# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F\n# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392\n# including the bash line at the top, and all but the python3 commands\n")
    
    commands.append("\n#SBATCH -N 1\n")                         
    commands.append("#SBATCH -p cpl\n")
    commands.append(f"#SBATCH --gres=gpu:{2 if two_gpus else 1}\n")
    commands.append(f"#SBATCH -t {time_alloc_hrs_str}\n")
    commands.append(f"#SBATCH --mem={mem_alloc_gb}G\n")
    commands.append("#SBATCH --constraint=high-capacity\n")
    commands.append(slurm_organization_command)
    
    commands.append(f"mkdir -p {slurm_folder}\n")
    
    commands.append("\nmodule load openmind/singularity/3.2.0\n")
    
    return commands

# end taken code
    