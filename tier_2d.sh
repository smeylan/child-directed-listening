#!/bin/bash

# 2d eval the child-specific models

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=9G
#SBATCH --output=./%j_gen-scripts_2c.out 

module load openmind/singularity/3.2.0

rm -rf output/SLURM/scripts_child_eval

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 src/gen/gen_child_eval_scripts.py

chmod u+x ./output/SLURM/submission_scripts/submit_child_eval.sh

./output/SLURM/submission_scripts/submit_child_eval.sh
