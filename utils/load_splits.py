# Code for loading the training data that has been split.


import os
from os.path import join, exists
 
from utils import split_gen
import glob



def 
##################
## TEXT LOADING ##
##################


def load_splits_folder_text(split, base_dir):
    
    folders = glob.glob(base_dir+'/*') # List the child names
    
    data = {}
    for path in folders:
        name = path.split('/')[-1]
        data[name] = load_split_text_path(split, name, base_dir)
        
    return data


def load_split_text_path(split, dataset, base_dir):
    
    # What else is needed?
    
    train_text_path = join(split_gen.get_split_folder(split, dataset, base_dir), 'train.txt')
    val_text_path = join(split_gen.get_split_folder(split, dataset, base_dir), 'validation.txt')
    
    # For the analyses? Think about what is required for yyy analysis.
    
    return {'train': train_text_path, 'val': val_text_path}
    
    
def load_eval_data_all(split_name, dataset_name, base_dir):
    
    """
    Loading cached data relevant to the model scoring functions in yyy analysis.
    """
    
    phono_filename = 'pvd_utt_glosses_phono_cleaned_inflated.pkl'
    success_utts_filename = 'success_utts.csv'
    yyy_utts_filename = 'yyy_utts.csv'

    data_filenames = [phono_filename, success_utts_filename, yyy_utts_filename]
    this_folder_path = split_gen.get_split_folder(split_name, dataset_name, base_dir)
    
    data_name = {
       'pvd_utt_glosses_phono_cleaned_inflated.pkl' : 'phono',
       'success_utts.csv' : 'success_utts',
       'yyy_utts.csv' : 'yyy_utts',
    }
    
    return {data_name[f] : pd.read_csv(join(this_folder_path, f)) for f in filenames }

    
    
    
    
