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
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/child/Lily/%j_training_beta_tags=True.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/child/Lily

module load openmind/singularity/3.2.0

rm -r /om2/user/wongn/child-directed-listening/experiments/no_versioning/models/child/Lily

rsync -a --progress /om2/user/wongn/child-directed-listening/experiments/no_versioning/models/all/all/with_tags /om2/user/wongn/child-directed-listening/experiments/no_versioning/models/child/Lily
mkdir ~/.cache/$SLURM_JOB_ID
# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu     python3 run_mlm.py             --model_name_or_path bert-base-uncased             --do_train             --do_eval             --output_dir /om2/user/wongn/child-directed-listening/experiments/no_versioning/models/child/Lily/with_tags            --train_file /om2/user/wongn/child-directed-listening/finetune/child/Lily/train.txt             --validation_file /om2/user/wongn/child-directed-listening/finetune/child/Lily/val.txt             --cache_dir ~/.cache/$SLURM_JOB_ID; singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu  python3 run_beta_search.py --split child --dataset Lily --context_width 0 --use_tags True --model_type childes
