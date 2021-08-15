#!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top, and all but the python3 commands

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --mem=13G
#SBATCH --constraint=high-capacity
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Alex/%j_data_Alex_prior_Violet.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Alex

module load openmind/singularity/3.2.0
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 run_child_cross.py --data_child Alex --prior_child Violet
# end all cites