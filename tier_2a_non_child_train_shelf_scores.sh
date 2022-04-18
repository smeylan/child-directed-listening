#!/bin/bash


#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=./%j_gen-scripts_2a.out 

export TOKENIZERS_PARALLELISM=true # Possibly only needed for wandb, but adding to avoid warning in case.

module load openmind/singularity/3.2.0

# scripts for nonchild, scripts for shelf models, scripts for finetune models

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_training_scripts.py & singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_eval_scripts.py; chmod u+x output/SLURM/submission_scripts/submit_non_child_train.sh & chmod u+x output/SLURM/submission_scripts/submit_non_child_beta_time_shelf.sh;

# train non-child models + score shelf models 

./output/SLURM/submission_scripts/submit_non_child_beta_time_shelf.sh 

# score shelf models

./output/SLURM/submission_scripts/submit_non_child_train.sh

