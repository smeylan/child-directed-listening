#!/bin/bash

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=9G
#SBATCH --output=./%j_gen-scripts_2b.out 

module load openmind/singularity/3.2.0

# gen scripts for children train

rm -rf output/SLURM/scripts_child_train

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_child_training_scripts.py

chmod u+x ./output/SLURM/submission_scripts/submit_non_child_beta_time_finetune.sh 
chmod u+x ./output/SLURM/submission_scripts/submit_child_train.sh

# score the fine-tuned models
./output/SLURM/submission_scripts/submit_non_child_beta_time_finetune.sh 
# train the child-specific models
./output/SLURM/submission_scripts/submit_child_train.sh
