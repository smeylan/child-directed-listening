
#!/bin/bash

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=/om2/user/wongn/child-directed-listening/%j_gen-scripts_2c.out 

module load openmind/singularity/3.2.0

singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 gen_child_eval_scripts.py

chmod u+x ./submit_child_cross.sh

./submit_child_cross.sh

# Then the following on Chompsky:
# tmux attach-session -t experiments
# rsync -a --progress wongn@openmind.mit.edu:~/child_repo_split/experiments ./experiments