#!/bin/bash


#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=1G
#SBATCH --output=./%j_gen-scripts_2a.out 

export TOKENIZERS_PARALLELISM=true # Possibly only needed for wandb, but adding to avoid warning in case.

module load openmind/singularity/3.2.0

# scripts for nonchild, scripts for shelf models, scripts for finetune models

# generate the training, fitting, and eval scripts 
singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_training_scripts.py; 
singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_fitting_scripts.py; 
singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_eval_scripts.py; 

# make the resulting scripts executable
chmod u+x output/SLURM/submission_scripts/submit_non_child_train.sh
chmod u+x output/SLURM/submission_scripts/submit_non_child_shelf_fit.sh
chmod u+x output/SLURM/submission_scripts/submit_non_child_unigram_fit.sh


./output/SLURM/submission_scripts/submit_non_child_train.sh
./output/SLURM/submission_scripts/submit_non_child_shelf_fit.sh
./output/SLURM/submission_scripts/submit_non_child_unigram_fit.sh

# need to wait until child training is done before doing finetune_fit



