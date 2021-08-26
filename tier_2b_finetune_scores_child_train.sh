#!/bin/bash

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=./%j_gen-scripts_2b.out 

module load openmind/singularity/3.2.0

# gen scripts for children train

rm -r scripts_child_train
rm -r scripts_child_cross

singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 gen_child_scripts.py

# train child models

chmod u+x ./submit_non_child_beta_time_finetune.sh & chmod u+x ./submit_child_train.sh

./submit_non_child_beta_time_finetune.sh & ./submit_child_train.sh
