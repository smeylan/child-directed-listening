
import os
from os.path import join, exists
from utils_child import child_models
from utils import scripts

import gen_sample_scripts

import configuration
config = configuration.Config()

if __name__ == '__main__':
    
    sh_loc = join(config.root_dir, 'scripts_child_cross')
    
    if not exists(sh_loc):
        os.makedirs(sh_loc)
    
    task_name = 'child_cross'
    all_names = child_models.get_child_names()
    
    scripts.gen_submit_script(task_name, [('child', name) for name in all_names], task_name)
    
    for data_child in all_names:
        for prior_child in all_names:
            
            slurm_folder = scripts.get_slurm_folder('child', data_child, 'child_cross') 
            
            command = f'python3 run_child_cross.py --data_child {data_child} --prior_child {prior_child}'
            
            with open(join(sh_loc, f'run_cross_{data_child}_{prior_child}.sh'), 'w') as f:
                
                time, mem = gen_sample_scripts.time_and_mem_alloc()
                
                headers = scripts.gen_command_header(mem_alloc_gb = mem, time_alloc_hrs = time,
                                                     slurm_folder = slurm_folder,
                                                     slurm_name = f'data_{data_child}_prior_{prior_child}') + [scripts.gen_singularity_header()]
                
                f.writelines(headers + [command] + ['\n# end all cites'])
         
     
    