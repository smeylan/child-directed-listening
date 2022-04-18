import os
from os.path import join, exists
import json
from datetime import datetime
from os.path import join, exists
import sys

sys.path.append('.')
sys.path.append('src/.')
from src.utils import scripts,  configuration, load_models, paths
config = configuration.Config()

def gen_training_commands(test_split, 
        training_split,
        test_dataset,
        training_dataset,
        model_type,
        use_tags,
        context_width,        
        task_name,
        task_phase,
        **kw):
            
    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = scripts.get_training_alloc(training_dataset)
    
    header_commands = scripts.gen_command_header(
        mem_alloc_gb = mem_alloc_gb, 
        time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        two_gpus = (training_dataset in {'Providence', 'Providence-Age'}))
    slurm_commands = header_commands

    # get the directory where this should be 
    model_output_dir = paths.get_directory(
        test_split, 
        training_split,
        test_dataset,
        training_dataset,
        model_type,
        use_tags,
        context_width,        
        task_name,
        task_phase)

    if not exists(model_output_dir):
        os.makedirs(model_output_dir)    
    
    slurm_commands += [f"rm -r {model_output_dir}\n"]  # clear the directory in case it had stuff in it before
    slurm_commands += [f"mkdir -p {model_output_dir}\n"]  # make the training directory if necessary     
    slurm_commands += ["mkdir ~/.cache/$SLURM_JOB_ID\n"]

    data_input_dir = paths.get_directory(
        test_split, 
        training_split,
        test_dataset,
        training_dataset,
        model_type,
        use_tags,
        context_width,        
        task_name,
        task_phase="data")    

    # Construct the python/training-related commands
    slurm_commands.append(scripts.get_run_mlm_command(training_split, training_dataset, use_tags,  data_input_dir, model_output_dir, config.slurm_user))
        
    slurm_filename = paths.get_slurm_script_path(
        test_split, 
        training_split,
        test_dataset,
        training_dataset,
        model_type,
        use_tags,
        context_width,        
        task_name,
        task_phase)

    return slurm_filename, slurm_commands

if __name__ == '__main__':
    
    task_name = 'non_child'    
    task_phase = 'train'
    
    finetune_models = load_models.gen_finetune_model_args() 
    # this includes +- context, whereas training doesn't care about context

    finetune_models_no_context = []
    for model in finetune_models:
        model['context_width'] = None #training doesn't manipulate context
        model['task_name'] = task_name
        model['test_split'] = None
        model['test_dataset'] = None # training datast is test dataset
        model['task_phase'] = task_phase        
        model['sh_loc'] = f'output/SLURM/{task_name}_{task_phase}'

        if not exists(model['sh_loc']):
            os.makedirs(model['sh_loc'])

        finetune_models_no_context.append(model)
    
    # take the unique models, collapsing across values of context
    unique_finetune_models = [dict(s) for s in set(frozenset(d.items()) for d in finetune_models_no_context)]

    for model in unique_finetune_models:

        fit_file, fit_commands = gen_training_commands(**model)

        with open(join(model['sh_loc'], fit_file), 'w') as f:
            f.writelines(fit_commands)
    

    scripts.gen_submit_script(task_name, task_phase)
