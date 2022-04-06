import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime


sys.path.append('.')
sys.path.append('src/.')
from src.utils import parsers, child_models, utils_child, split_gen, load_models


if __name__ == '__main__':
        
    start_time = str(datetime.today())
    parser = parsers.split_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]    
    this_model_args = vars(raw_args)
    this_model_args = vars(raw_args)
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    # end cites
       
    levdist_scores, beta_used = utils_child.score_cross_prior(this_model_args, 'levdist')
    wfst_scores, lambda_used = utils_child.score_cross_prior(this_model_args, 'wfst')
    scores =  pd.concat([levdist_scores, wfst_scores])
    
    score_path = utils_child.get_cross_path(this_model_args['dataset'], this_model_args['training_dataset'], this_model_args['model_type'])
    
    scores.to_pickle(score_path)
    
    print(f'Computations complete for: {this_model_args["dataset"]}, {this_model_args["training_dataset"]}')
    print(f'Scores saved to: {score_path}')
    
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
   