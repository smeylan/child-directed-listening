import config
import config_train

import json

import os
from os.path import join, exists

from pprint import pprint


def load_hyperparam_info(split, dataset, is_tags, lr):
    
    base_dir = join(config.root_dir, 'experiments/lr_search/models') 
    this_model_dir = join(base_dir, join(join(split, dataset), 'with_tags' if is_tags else 'no_tags'))
    this_dir = join(this_model_dir, f'{lr}')
    
    print("Temporarily skipping outputs that don't have eval_results.json.")
   
    this_path = join(this_dir, 'eval_results.json')
    
    if exists(this_path):
        with open(this_path, 'r') as f:
            eval_results = json.load(f)
            return eval_results
    else:
        return {'eval_loss' : -float('inf') } # Temporary only!

    
def find_best_lr(split, dataset, is_tags):
    
    all_lr = config_train.lr_search_params
    
    lr_results = {}
    
    for lr in all_lr:
        results = load_hyperparam_info(split, dataset, is_tags, lr)
        lr_results[str(lr)] = results['eval_loss']
    
    pprint(lr_results)
    
    argmax = lambda this_lr : lr_results[this_lr]
    best_lr = max([str(num) for num in all_lr], key = argmax)
    
    pprint(lr_results)
    
    print(f'Selected: {best_lr}')
    
    return float(best_lr), lr_results[best_lr]


def get_best_lr_dict_loc():
    
    base_dir = join(config.root_dir, 'experiments/lr_search/models') 
    this_path = join(base_dir, 'lr_search_results.json')
        
    return this_path

def gen_best_lr_dict():
    
    this_path = get_best_lr_dict_loc()
    
    lr_records = {}
    for arg_set in config.childes_model_args:
        for tags in [True, False]:
            
            split, dataset = arg_set
            this_lr = find_best_lr(split, dataset, tags)
            
            split_name = '/'.join(arg_set + ('with_tags' if tags else 'no_tags',))
            lr_records[split_name] = this_lr
    with open(this_path, 'w') as f:
        json.dump(lr_records, f)
        
    print(f'Hyperparameter analysis information written to: {this_path}')
                
        
if __name__ == '__main__':
    gen_best_lr_dict()
        
    
     
    