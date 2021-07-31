#!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top, and all but the python3 commands

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH --gres=gpu:1
#SBATCH -t 7:00:00
#SBATCH --mem=9G
#SBATCH --constraint=high-capacity

module load openmind/singularity/3.2.0
mkdir ~/.cache/$SLURM_JOB_ID
# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu     python3 run_mlm.py             --model_name_or_path bert-base-uncased             --do_train             --do_eval             --output_dir /om2/user/wongn/child-directed-listening/experiments/models/no_versioning/age/young/no_tags            --train_file /home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening/finetune/age/young/train_no_tags.txt             --validation_file /home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening/finetune/age/young/val_no_tags.txt             --cache_dir ~/.cache/$SLURM_JOB_ID             --overwrite_output_dir
# end taken command code 6/24/21 and slurm id reference 7/13/21