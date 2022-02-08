#!/bin/bash

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=9G
#SBATCH --output=./%j_gen-scripts_2c.out 

module load openmind/singularity/3.2.0

rm -r scripts_child_cross

singularity exec --nv -B /om,/om2/user/${CDL_SLURM_USER} ${CDL_SINGULARITY_PATH} python3 gen_child_eval_scripts.py

chmod u+x ./submit_child_cross.sh

./submit_child_cross.sh
