import copy
import sys
import os

sys.path.append('.')
sys.path.append('src/.')
from src.utils import paths, configuration, scripts

config = configuration.Config()

def gen_training_commands(spec_dict):


    paths.validate_spec_dict(spec_dict, config.spec_dict_params)
    paths.validate_phase(spec_dict['task_phase'], config.task_phases)

    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = scripts.get_training_alloc(spec_dict['training_dataset'])
    
    header_commands = scripts.gen_command_header(
        mem_alloc_gb = mem_alloc_gb, 
        time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        two_gpus = (spec_dict['training_dataset'] in {'Providence', 'Providence-Age'}))
    slurm_commands = header_commands

    # get the directory where this should be saved
    model_output_spec_dict  = copy.copy(spec_dict)
    model_output_spec_dict['task_phase'] = 'train'

    model_output_dir = paths.get_directory(model_output_spec_dict)    

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)    
    
    slurm_commands += [f"rm -rf {model_output_dir}\n"]  # clear the directory in case it had stuff in it before. f in case it doesn't exist
    slurm_commands += [f"mkdir -p {model_output_dir}\n"]  # make the training directory if necessary     
    slurm_commands += ["mkdir ~/.cache/$SLURM_JOB_ID\n"]

    data_input_spec_dict  = copy.copy(spec_dict)
    data_input_spec_dict['task_phase'] = 'extract_data'

    data_input_dir = paths.get_directory(data_input_spec_dict)        

    sh_loc = 'output/SLURM/'+spec_dict['task_name']+'_'+spec_dict['task_phase']

    if not os.path.exists(sh_loc):
        os.makedirs(sh_loc)

    # Construct the python/training-related commands
    if spec_dict['model_type'] == 'BERT':
        slurm_commands.append(scripts.get_run_mlm_command(spec_dict, data_input_dir, model_output_dir, config.slurm_user))
    
    elif spec_dict['model_type'] == 'GPT-2':
        slurm_commands.append(scripts.get_run_clm_command(spec_dict, data_input_dir, model_output_dir, config.slurm_user))    


    slurm_filename = os.path.join(sh_loc, paths.get_slurm_script_name(spec_dict))
    print(slurm_filename)
    
    return slurm_filename, slurm_commands