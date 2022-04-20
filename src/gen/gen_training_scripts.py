import os
from os.path import join, exists
import json
import copy
from datetime import datetime
from os.path import join, exists
import sys

sys.path.append('.')
sys.path.append('src/.')
from src.utils import scripts,  configuration, load_models, paths
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

    if not exists(model_output_dir):
        os.makedirs(model_output_dir)    
    
    slurm_commands += [f"rm -r {model_output_dir}\n"]  # clear the directory in case it had stuff in it before
    slurm_commands += [f"mkdir -p {model_output_dir}\n"]  # make the training directory if necessary     
    slurm_commands += ["mkdir ~/.cache/$SLURM_JOB_ID\n"]

    data_input_spec_dict  = copy.copy(spec_dict)
    data_input_spec_dict['task_phase'] = 'extract_data'

    data_input_dir = paths.get_directory(data_input_spec_dict)        

    sh_loc = 'output/SLURM/'+spec_dict['task_name']+'_'+spec_dict['task_phase']

    if not exists(sh_loc):
        os.makedirs(sh_loc)

    # Construct the python/training-related commands
    slurm_commands.append(scripts.get_run_mlm_command(
        spec_dict['training_split'], 
        spec_dict['training_dataset'], 
        spec_dict['use_tags'],
        data_input_dir, model_output_dir, config.slurm_user))
        
    slurm_filename = os.path.join(sh_loc, paths.get_slurm_script_name(spec_dict))
    
    return slurm_filename, slurm_commands

if __name__ == '__main__':
    
    task_name = 'non_child'    
    task_phase = 'train'
    
    finetune_models = load_models.gen_finetune_model_args() 
    # this includes +- context, whereas training doesn't care about context

    finetune_models_no_context = []
    for model in finetune_models:        
        for use_tags in (True, False):
            model['context_width'] = None #training doesn't manipulate context
            model['task_name'] = task_name
            model['test_split'] = None
            model['test_dataset'] = None # training datast is test dataset
            model['task_phase'] = task_phase
            model['use_tags'] = use_tags
            model['n_samples'] = config.n_across_time                

            finetune_models_no_context.append(model)
    
    # take the unique models, collapsing across values of context
    unique_finetune_models = [dict(s) for s in set(frozenset(d.items()) for d in finetune_models_no_context)]

    for model in unique_finetune_models:

        fit_file, fit_commands = gen_training_commands(model)

        with open(join(fit_file), 'w') as f:
            f.writelines(fit_commands)
    

    scripts.gen_submit_script(task_name, task_phase)
