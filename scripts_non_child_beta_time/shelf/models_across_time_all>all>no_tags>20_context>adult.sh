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
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/all/all/%j_non_child_beta_time_model=adult_tags=False_context=20.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/all/all

module load openmind/singularity/3.2.0
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 run_beta_search.py --split all --dataset all --context_width 20 --use_tags False --model_type adult; singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu  python3 run_models_across_time.py --split all --dataset all --context_width 20 --use_tags False --model_type adult