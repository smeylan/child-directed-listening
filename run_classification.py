
#import argparse
from utils_model_analysis import classify_speaker

import time

import config

if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('split', type = str, help = 'Which split to use. childes: {all, age}. All others use "all" split.')
#     parser.add_argument('dataset', type = str, help = 'Which sub-split to use. childes/all, any other models: {all}. childes/age: {old, young}')
     
#     args = vars(parser.parse_args())
#     split = args['split']; dataset = args['dataset']

    start_time = time.time()
    last_time = start_time
    
    for split, dataset in config.childes_model_args:
    
        classify_speaker.analyze_model_tags(split, dataset)
        
        current_time = time.time()
        print(f"Processed {split}, {dataset}")
        print(f'Time since start (in minutes): {(current_time - start_time) / 60.0}')
        print(f'Time since last completion (in minutes): {(current_time - last_time) / 60.0}')
        
        last_time = current_time