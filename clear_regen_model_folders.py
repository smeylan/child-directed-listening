import os
from os.path import join, exists

import shutil
import write_training_scripts

if __name__ == '__main__':
    
    om2_user = 'wongn'
    base_dir = f'/om2/user/{om2_user}/childes_run'
    
    all_splits = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    for split_args in all_splits:
        for has_tags in [True, False]:
            
            t_split, t_dataset = split_args
            this_model_dir = write_training_scripts.models_get_split_folder(t_split, t_dataset, has_tags, base_dir)
            
            if exists(this_model_dir):
                print(f'Clearing folder {this_model_dir}')
                shutil.rmtree(this_model_dir)
                os.makedirs(this_model_dir) 
