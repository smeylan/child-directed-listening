# Permits quick small samples to allow for Chompsky local development with realistic args
# This is for development only, not for results.


import subprocess 
from utils import load_models


if __name__ == '__main__':
    
    
    model_args = load_models.gen_all_model_args()
    
    task_names = ['beta_search']#, 'models_across_time']
    task_files = ['run_beta_search.py'] #, 'run_models_across_time.py']
    
    for task_name, task_file in zip(task_names, task_files):
        
        
         for arg_set in model_args:

            split, dataset, use_tags, context_width, model_type = arg_set

            model_id = load_models.get_model_id(
                    split, dataset, use_tags, context_width, model_type
                ).replace('/', '>')
            command = f"python3 {task_file} --split {split} --dataset {dataset} --context_width {context_width} --use_tags {use_tags} --model_type {model_type}" # This may have to be "python3" on openmind? 


            print(f'Processing: task: {task_name}, model: {model_id}')
            # 7/10/21: https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output
            # 7/10/21: https://docs.python.org/3/library/subprocess.html
            subprocess.run(command, shell=True, check=True)


    