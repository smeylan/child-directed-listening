import os
import sys
from os.path import join, exists
sys.path.append('.')
sys.path.append('src/.')


sys.path.append('.')
sys.path.append('src/.')
from src.gen import gen_training_scripts, gen_sample_scripts
from src.utils import split_gen, scripts, configuration, child_models, fitting
config = configuration.Config()


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
                    'context_width':20,
                    'task_name': task_name})

    
    # Pretends that Switchboard and CDL+Context are kids

    for data_child in child_names:
        child_arg_list.append(
            {'split_name': 'child', 'dataset_name': data_child, 'training_split_name':'switchboard', 'training_dataset_name': 'all', 'model_type':'switchboard', 'use_tags':False,
            'context_width':20,
            'task_name': task_name})

    for data_child in child_names:
        child_arg_list.append(
            {'split_name': 'child', 'dataset_name': data_child, 'training_split_name':'all', 'training_dataset_name': 'all', 'model_type':'childes', 
            'use_tags':True,
            'context_width':20,
            'task_name': task_name})
    
    # then I might need another gen_submit_script 

    if not exists(sh_fit_loc):
        os.makedirs(sh_fit_loc)

    for child_arg in child_arg_list:
        fit_file, fit_commands = fitting.gen_fitting_commands(**child_arg)

        with open(join(sh_fit_loc, fit_file), 'w') as f:
                f.writelines(fit_commands)


    scripts.gen_submit_script(task_name, child_arg_list, 'child_fit')
    