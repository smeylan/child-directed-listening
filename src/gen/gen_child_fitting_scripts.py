import os
import sys
from os.path import join, exists
sys.path.append('.')
sys.path.append('src/.')


sys.path.append('.')
sys.path.append('src/.')
from src.gen import gen_training_scripts, gen_sample_scripts
from src.utils import split_gen, scripts, configuration, child_models
config = configuration.Config()


def gen_child_fitting_commands(
        split_name,
        training_split_name, 
        dataset_name,
        training_dataset_name,
        model_type,
        use_tags,
        context_width):
    
    your_model_path = split_gen.get_split_folder('child', training_dataset_name, config.model_dir)
    
    # ---------- begin new code
    
    # Generate the appropriate header and the slurm folder
    
    slurm_folder = scripts.get_slurm_folder(split_name, training_dataset_name, 'child_train')
    
    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = gen_training_scripts.get_training_alloc('child')
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        slurm_folder = slurm_folder,
        slurm_name = f'training_beta_tags={use_tags}', 
        two_gpus = False)
    commands = header_commands


    this_model_dir = '/'.join(gen_training_scripts.models_get_split_folder(split_name, training_dataset_name, use_tags).split('/')[:-1])
    

    sing_header = scripts.gen_singularity_header()   
    
    run_commands = [f"{sing_header} {gen_sample_scripts.get_one_python_command('src/run/run_beta_search.py', split_name, dataset_name , use_tags, context_width, model_type, training_dataset_name, training_split_name)[1]}\n"]    
        
    # Put the copy commands between the header and the actual python runs.
    commands += run_commands
    
    filename = scripts.get_script_name('child', training_dataset_name, use_tags, dataset_name, model_type)

    return filename, commands


if __name__ == '__main__':
    
    _, is_tags = child_models.get_best_child_base_model_path()
    child_names = child_models.get_child_names()
    
    task_name = 'child_fit'
    
    sh_fit_loc = f'output/SLURM/scripts_{task_name}'

    child_arg_list = []

    # full cross of the child models
    for model_child in child_names:  
        for data_child in child_names:
            child_arg_list.append(
                {'split_name': 'child', 'dataset_name': data_child, 'training_split_name':'child', 'training_dataset_name': model_child, 'model_type':'childes', 'use_tags':True,
                    'context_width':20})

    
    # Pretends that Switchboard and CDL+Context are kids

    for data_child in child_names:
        child_arg_list.append(
            {'split_name': 'child', 'dataset_name': data_child, 'training_split_name':'switchboard', 'training_dataset_name': 'all', 'model_type':'switchboard', 'use_tags':False,
            'context_width':20})

    for data_child in child_names:
        child_arg_list.append(
            {'split_name': 'child', 'dataset_name': data_child, 'training_split_name':'all', 'training_dataset_name': 'all', 'model_type':'childes', 
            'use_tags':True,
            'context_width':20})
    
    # then I might need another gen_submit_script 

    if not exists(sh_fit_loc):
        os.makedirs(sh_fit_loc)

    for child_arg in child_arg_list:
        fit_file, fit_commands = gen_child_fitting_commands(**child_arg)

        with open(join(sh_fit_loc, fit_file), 'w') as f:
                f.writelines(fit_commands)


    scripts.gen_submit_script(task_name, child_arg_list, 'child_fit')
    