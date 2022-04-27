import os
import sys
from os.path import join, exists

sys.path.append('.')
sys.path.append('src/.')
from src.utils import split_gen, scripts, configuration, child_models, fitting, load_models, paths, evaluation  
config = configuration.Config()


if __name__ == '__main__':

    task_phase = 'eval'
    task_name = 'child'

    child_names = child_models.get_child_names()


    # finetune_models = load_models.gen_finetune_model_args()
    # shelf_models = load_models.gen_shelf_model_args() 
    # unigram_models = load_models.gen_unigram_model_args() 
    # child_model_set = load_models.gen_child_model_args()
    
    # full cross of the child models

    child_arg_list = []
    for training_child in child_names:      
        for test_child in child_names:
            child_arg_list.append(
                {'training_split': 'Providence-Child',
                'training_dataset': training_child,
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':True,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})

    
    # Pretends that Switchboard is a kid
    for test_child in child_names:
        child_arg_list.append(
            {'training_split': 'Switchboard',
                'training_dataset': 'all',
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':False,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})


    # Pretends that Switchboard is a kid
    for test_child in child_names:
        child_arg_list.append(
            {'training_split': 'Providence',
                'training_dataset': 'all',
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':True,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})

    sh_fit_loc = f'output/SLURM/{task_name}_{task_phase}'
    if not exists(sh_fit_loc):
        os.makedirs(sh_fit_loc) 

    for child_arg in child_arg_list:
        fit_file, fit_commands = evaluation.gen_evaluation_commands(child_arg)

        with open(fit_file, 'w') as f:
                f.writelines(fit_commands)


    scripts.gen_submit_script(task_name, task_phase)
    