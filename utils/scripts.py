

        
 
    
# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

    
def gen_singularity_header(om2_user = 'wongn'):
    return f"singularity exec --nv -B /om,/om2/user/{om2_user} /om2/user/{om2_user}/vagrant/trans-pytorch-gpu " 
    

def gen_command_header(time_alloc_hrs):
    
    commands = []
    commands.append("#!/bin/bash\n")
    
    # Citation text for every script
    commands.append("\n# For the command text\n# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F\n# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392\n# including the bash line at the top\n")
    
    commands.append("\n#SBATCH -N 1\n")                         
    commands.append("#SBATCH -p cpl\n")
    commands.append("#SBATCH --gres=gpu:1\n")
    commands.append(f"#SBATCH -t {time_alloc_hrs}:00:00\n")
    commands.append("#SBATCH --mem=9G\n")
    commands.append("#SBATCH --constraint=high-capacity\n")
     
    commands.append("\nmodule load openmind/singularity/3.2.0\n")
    
    return commands
    