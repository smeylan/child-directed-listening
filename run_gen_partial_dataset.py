
# For getting models to converge within memory constraints/faster iteration time.

import config
import config_train

import random
import os
from os.path import join, exists

from utils import split_gen

random.seed(config.SEED)

def cut_texts(split, dataset, save_dir, phase):
    
    orig_folder = split_gen.get_split_folder(split, dataset, config.finetune_dir)
    orig_path = join(orig_folder, f'{phase}.txt')
    
    with open(orig_path, 'r') as f:
        text = f.readlines()
    
    random.shuffle(text)
    num_samples = int(config_train.cut_ratio * len(text)) if not isinstance(config_train.cut_ratio, int) else config_train.cut_ratio
    
    cut_text = text[:num_samples]
    
    no_tags_cut_text = split_gen.filter_text_from_content(cut_text)
    
    # Write the results 
    
    new_folder = split_gen.get_split_folder(split, dataset, save_dir)
    
    with_tags_path = join(new_folder, f'{phase}.txt')
    no_tags_path = join(new_folder, f'{phase}_no_tags.txt')
    
    with open(with_tags_path, 'w') as f:
        f.writelines(cut_text)
        
    with open(no_tags_path, 'w') as f:
        f.writelines(no_tags_cut_text) 
    
    print(f'Wrote subsampled files to:')
    print(f'\t{with_tags_path}')
    print(f'\t{no_tags_path}')
    
    
if __name__ == '__main__':
   
    cut_base_dir = join(config.root_dir, config_train.finetune_cut_dir_name)
    
    for split, dataset in config.childes_model_args:
        for phase in ['train', 'val']:
            cut_texts(split, dataset, cut_base_dir, phase)
    
    