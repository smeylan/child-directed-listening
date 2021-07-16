

import os
from os.path import join, exists

from utils_child import child_models
import gen_training_scripts


if __name__ == '__main__':
    
    child_names = child_models.get_child_names()
    # Get the optimal configuration from the 
    
    for child in child_names:
        # Copy their model base over in preparation for the beta optimization.
        # Will also generate the model folders.
        this_model_path, is_tags = child_models.gen_finetuning_base(child)
        
    