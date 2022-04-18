import os
import sys
import copy
from os.path import join, exists
sys.path.append('.')
sys.path.append('src/.')


sys.path.append('.')
sys.path.append('src/.')
from src.gen import gen_training_scripts, gen_eval_scripts
from src.utils import split_gen, scripts, configuration, child_models, fitting, load_models
config = configuration.Config()


if __name__ == '__main__':
            
    for task_name in ['non_child_fit_shelf','non_child_fit_finetune']:
        sh_fit_loc = f'output/SLURM/scripts_{task_name}'
        if not exists(sh_fit_loc):
            os.makedirs(sh_fit_loc)


    finetune_models = load_models.gen_finetune_model_args()
    shelf_models = load_models.gen_shelf_model_args() 

    partitions = {
        'finetune' : finetune_models,
        'shelf' : shelf_models,
    }    

    sh_fit_loc = f'output/SLURM/scripts_non_child_fit_shelf'
    shelf_model_args = []
    for model in partitions['shelf']:            
        model['task_name'] = 'non_child_fit_shelf'
        model['training_dataset_name'] = model['dataset_name']
        model['training_split_name'] = model['split_name']
        shelf_model_args.append(copy.copy(model))
    
    
    for shelf_arg in shelf_model_args:
        fit_file, fit_commands = fitting.gen_fitting_commands(**shelf_arg)

        with open(join(sh_fit_loc, fit_file), 'w') as f:
                f.writelines(fit_commands)

    scripts.gen_submit_script('non_child_fit_shelf', shelf_arg, 'non_child_fit_shelf')

    sh_fit_loc = f'output/SLURM/scripts_non_child_fit_finetune'
    finetune_model_args = []
    for model in partitions['finetune']:         
        model['task_name'] = 'non_child_fit_finetune'
        model['training_dataset_name'] = model['dataset_name']
        model['training_split_name'] = model['split_name']           
        finetune_model_args.append(copy.copy(model))                        
    

    for finetune_arg in finetune_model_args:
        fit_file, fit_commands = fitting.gen_fitting_commands(**finetune_arg)

        with open(join(sh_fit_loc, fit_file), 'w') as f:
                f.writelines(fit_commands)

    scripts.gen_submit_script('non_child_fit_finetune', finetune_arg, 'non_child_fit_finetune')