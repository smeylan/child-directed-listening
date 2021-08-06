#!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top, and all but the python3 commands

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH --gres=gpu:1
#SBATCH -t 0:45:00
#SBATCH --mem=13G
#SBATCH --constraint=high-capacity
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/name/William/%j_data_William_prior_Violet.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/name/William

module load openmind/singularity/3.2.0
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 run_child_cross.py --data_child William --prior_child Violet
# end all cites