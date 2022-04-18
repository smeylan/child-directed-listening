import glob
import os
from os.path import join, exists
import subprocess

from src.utils import split_gen, configuration
config = configuration.Config()

def get_slurm_folder(split, dataset, task):

    raise ValueError('Deprecated')
    
    base_paths = {
        'non_child_train' : config.model_dir, # Non-child train
        'non_child_fit_finetune' : config.model_dir, # Non-child train
        'non_child_fit_shelf' : config.model_dir,
        'non_child_eval' : config.scores_dir, # Non-child beta + time scoring
        
        'child_train' : config.scores_dir, # Child train 
        'child_fit' : config.scores_dir, # Child beta and lambda fitting
        'child_eval' : config.scores_dir, # Child scoring
    }
    assert task in base_paths.keys()
    return split_gen.get_split_folder(split, dataset, base_paths[task])
    
    
def get_slurm_folders_by_args(args, task):
    
    raise ValueError('Deprecated')


    all_paths = []
    
    for this_args in args:
        this_split = this_args['split_name']
        this_dataset = this_args['dataset_name']
        this_path = get_slurm_folder(this_split, this_dataset, task)
        all_paths.append(this_path)
    
    return sorted(list(set(all_paths)))
    
def gen_submit_script(task_name, task_phase):
    
    text = ['#!/bin/bash -e']
        
    base_dir = f'./output/SLURM/{task_name}_{task_phase}'
    text.append(f'FILES="{base_dir}/*"')
    text.append('for f in $FILES')
    text.append('do')
    text.append('\techo "Processing $f file..."')
    text.append('\tsbatch $f')
    text.append('\tcat "$f"')
    text.append('done')
    
    
    submit_sh_path = f'output/SLURM/submission_scripts/submit_{task_name}_{task_phase}.sh' 
    
    # make sure that the path exists
    submit_sh_dir = os.path.dirname(submit_sh_path)
    if not os.path.exists(submit_sh_dir):
        os.makedirs(submit_sh_dir)

    
    give_space = lambda s : f"{s}\n"
    text = list(map(give_space, text))
    
    with open(submit_sh_path, 'w') as f:
        f.writelines(text)
    
    subprocess.call(f'chmod u+x {submit_sh_path}', shell = True)
    
    return text
    
    
def write_training_shell_script(split,  dataset,  is_tags, context, training_split, dir_name, training_dataset, get_command_func, om2_user = config.slurm_user): 
    
    if not exists(dir_name):
        os.makedirs(dir_name)
    
    script_name = get_script_name(split, dataset, is_tags, context, training_dataset, training_split)
    
    with open(join(dir_name, script_name), 'w') as f:
        f.writelines(get_command_func(split, dataset, is_tags, om2_user = om2_user))
        



    
    
    

    
# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

    
def gen_singularity_header(om2_user = config.slurm_user):
    
    # still part of the taken code above
    return f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/ubuntu20.simg " 
    
def format_time(args):
    
    # Skip formatting hours, it should not add a 0 in front of the hours.
    # If a 0 is added it will terminate running early
     
    new_args = tuple([f'0{arg}' if arg < 10 else str(arg) for arg in args[1:]])
    return (args[0],) + new_args
  

def gen_command_header(mem_alloc_gb, time_alloc_hrs, n_tasks, cpus_per_task, two_gpus = False):

    slurm_folder = config.slurm_log_dir
    
    if isinstance(time_alloc_hrs, int):
        time_alloc_hrs_str = f'{time_alloc_hrs}:00:00'
    if isinstance(time_alloc_hrs, tuple):
        hrs, mins, secs = format_time(time_alloc_hrs)
        time_alloc_hrs_str = f'{hrs}:{mins}:{secs}'
              
    
    slurm_organization_command = f"#SBATCH --output={slurm_folder}/%j.out\n"
    
    commands = []
    commands.append("#!/bin/bash -e\n")
    
    # Citation text for every script
    commands.append("\n# For the command text\n# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F\n# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392\n# including the bash line at the top, and all but the python3 commands\n")
    
    commands.append("\n#SBATCH -N 1\n")                         
    commands.append("#SBATCH -p cpl\n")
    commands.append(f"#SBATCH --gres=gpu:{2 if two_gpus else 1}\n")
    commands.append(f"#SBATCH -t {time_alloc_hrs_str}\n")
    commands.append(f"#SBATCH --mem={mem_alloc_gb}G\n")
    commands.append("#SBATCH --constraint=high-capacity\n")

    commands.append(f"#SBATCH --ntasks={n_tasks}\n")
    commands.append(f"#SBATCH --cpus-per-task={cpus_per_task}\n")

    commands.append(slurm_organization_command)
    
    commands.append(f"mkdir -p {slurm_folder}\n")
    
    commands.append("\nmodule load openmind/singularity/3.2.0\n")
    
    return commands

# end taken code
    

def time_and_mem_alloc():
    
    is_subsample = (config.n_subsample <= 500) # Always use n_subsample, just depends if 500 or 1000
    
    this_time_alloc = (0, 10, 0) if config.dev_mode else ((1, 0, 0) if is_subsample else (12, 0, 0))
    this_mem_amount = 10 if config.dev_mode else (13 if is_subsample else 35)
    this_n_tasks = 1
    this_cpus_per_task = 24 
    
    return this_time_alloc, this_mem_amount, this_n_tasks, this_cpus_per_task


def get_training_alloc(training_dataset):
        
    time, mem, n_tasks, cpus_per_task = time_and_mem_alloc()
    if training_dataset != 'Providence-Child':
        time = 24 if not config.dev_mode else (0, 30, 0)                
    
    return mem, time, n_tasks, cpus_per_task



def get_run_mlm_command(training_split, training_dataset, use_tags, data_input_dir, model_output_dir, slurm_user):
    
    this_args_dict = config.child_args if training_split == 'Providence-Child' else config.general_training_args
    
    if training_split == 'Providence-Child':
        _, is_tags = child_models.get_best_child_base_model_path()
        base_model = models_get_split_folder('all', 'all', is_tags)
    else:
        base_model = 'bert-base-uncased'
        
    this_args_dict['model_name_or_path'] = base_model
    
    this_args_list = sorted(list(this_args_dict.keys())) # readability
    
    data_args = [
            f"--train_file {data_input_dir}/train.txt",
            f"--validation_file {data_input_dir}/val.txt", 
            f"--cache_dir ~/.cache/$SLURM_JOB_ID",
            f"--output_dir {model_output_dir}",
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

    main_command = f"singularity exec --nv -B /om,/om2/user/{slurm_user} /om2/user/{slurm_user}/vagrant/ubuntu20.simg"
    this_python_command = f' python3 src/run/run_mlm.py {" ".join(data_args + trainer_args)}'
    
    return f"{main_command}{this_python_command}"

