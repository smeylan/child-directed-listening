#!/bin/bash


#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=${CDL_SLURM_ROOT}/%j_gen-scripts_2a.out 

export TOKENIZERS_PARALLELISM=true # Possibly only needed for wandb, but adding to avoid warning in case.

module load openmind/singularity/3.2.0

# scripts for nonchild, scripts for shelf models, scripts for finetune models

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 gen_training_scripts.py & singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 gen_sample_scripts.py; chmod u+x submit_non_child_train.sh & chmod u+x submit_non_child_beta_time_shelf.sh;

# train non-child models + score shelf models 

${CDL_SLURM_ROOT}/submit_non_child_beta_time_shelf.sh & ${CDL_SLURM_ROOT}/x

# If in development: At this point, rsync models/scores to Chompsky