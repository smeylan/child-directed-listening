#!/bin/bash

# 2b: fit the finetune models, eval the shelf and unigram models, and train the child-specific models


#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=9G
#SBATCH --output=./output/logs/%j_gen-scripts_2b.out 

module load openmind/singularity/3.2.0

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_child_training_scripts.py

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_child_fitting_scripts.py

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_child_eval_scripts.py


chmod u+x ./output/SLURM/submission_scripts/submit_non_child_finetune_fit.sh
chmod u+x ./output/SLURM/submission_scripts/submit_non_child_shelf_eval.sh
chmod u+x ./output/SLURM/submission_scripts/submit_non_child_unigram_eval.sh
chmod u+x ./output/SLURM/submission_scripts/submit_child_train.sh

./output/SLURM/submission_scripts/submit_non_child_finetune_fit.sh
./output/SLURM/submission_scripts/submit_non_child_shelf_eval.sh
./output/SLURM/submission_scripts/submit_non_child_unigram_eval.sh
./output/SLURM/submission_scripts/submit_child_train.sh