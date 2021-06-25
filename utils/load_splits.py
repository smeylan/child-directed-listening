# Code for loading the training data that has been split.


import os
from os.path import join, exists
 
from utils import split_gen
import glob

def load_splits_folder_text(split, base_dir):
    
    # The ways that the folders are named are a little misleading...
    # base_dir should be the true base dir?
    
    folders = glob.glob(base_dir+'/*') # List the child names
    
    data = {}
    for path in folders:
        name = path.split('/')[-1]
        data[name] = load_split_text_path(split, name, base_dir)
        
    return data


def load_split_text_path(split, dataset, base_dir):
    
    # What else is needed?
    
    train_text_path = join(get_split_folder(split, dataset, base_dir), 'train.txt')
    val_text_path = join(get_split_folder(split, dataset, base_dir), 'validation.txt')
    
    # For the analyses? Think about what is required for yyy analysis.
    
    return train_text_path, val_text_path
    
    
    
    
