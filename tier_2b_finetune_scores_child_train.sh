
#!/bin/bash

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=/om2/user/wongn/child-directed-listening/%j_gen-scripts_2a.out 

module load openmind/singularity/3.2.0

# gen scripts for children and children cross + score finetune models

singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 gen_child_scripts.py & singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 gen_child_eval_scripts.py;

# train child models

chmod u+x ./submit_non_child_beta_time_finetune.sh & chmod u+x ./submit_child_train.sh & chmod u+x 
./submit_child_cross.sh;

./submit_non_child_beta_time_finetune.sh & ./submit_child_train.sh
./submit_child_cross.sh



# Then the following on Chompsky:
# tmux attach-session -t experiments
# rsync -a --progress wongn@openmind.mit.edu:~/child_repo_split/experiments ./experiments