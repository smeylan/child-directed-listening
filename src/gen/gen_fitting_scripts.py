import os
import sys
import copy
from os.path import join, exists
sys.path.append('.')
sys.path.append('src/.')
import copy

from src.gen import gen_training_scripts, gen_eval_scripts
from src.utils import split_gen, scripts, configuration, load_models, paths, fitting
config = configuration.Config()



if __name__ == '__main__':
    
    task_phase = 'fit'
    task_name = 'non_child'            


    finetune_models = load_models.gen_finetune_model_args()
    shelf_models = load_models.gen_shelf_model_args() 
    unigram_models = load_models.gen_unigram_model_args() 

    partitions = {
        'finetune' : finetune_models,
        'shelf' : shelf_models,
        'unigram': unigram_models
    }    
    
    for subtask in ['shelf', 'finetune', 'unigram']:

        subtask_name = task_name + '_' + subtask
        sh_fit_loc = f'output/SLURM/{subtask_name}_{task_phase}'
        if not exists(sh_fit_loc):
            os.makedirs(sh_fit_loc) 
        
        model_args = []
        for model in partitions[subtask]:            
            model['task_name'] = subtask_name
            model['test_split'] = 'Providence'
            model['test_dataset'] = 'all'       
            model['task_phase'] = task_phase
            model['n_samples'] = config.n_across_time                
            model_args.append(copy.copy(model))
        
        for arg_set in model_args:
            fit_file, fit_commands = fitting.gen_fitting_commands(arg_set)

            with open(fit_file, 'w') as f:
                f.writelines(fit_commands)

        scripts.gen_submit_script(subtask_name, task_phase)
