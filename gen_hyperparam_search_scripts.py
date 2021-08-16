

import config
import config_train
from utils import scripts
import gen_training_scripts

import os
from os.path import join, exists


if __name__ == '__main__':
    
    
    # Need to manually change config_train to use a different cut_ratio to generate the files
    # or use this function to generate the right scripts
    
    label = 'non_child_train_search'
    
    all_splits = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    for split_args in all_splits:
        for has_tags in [True, False]:
            for lr in config_train.lr_search_params:
                
                t_split, t_dataset = split_args
                tags_str = 'with_tags' if has_tags else 'no_tags'
                scripts.write_training_shell_script(t_split, t_dataset, has_tags, f'scripts_{label}/{tags_str}', gen_training_scripts.get_isolated_training_commands)
                
    # You actually need to write these sucessively? How to do this?
    # Or, submit separate searches (probably not as good?)

        scripts.gen_submit_script(label, config.childes_model_args, label) # How to submit the script properly?
    
    
    