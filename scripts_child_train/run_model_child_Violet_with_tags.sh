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
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Violet/%j_training_beta_tags=True.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Violet

module load openmind/singularity/3.2.0
rm -r /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/models/child/Violet
mkdir ~/.cache/$SLURM_JOB_ID
# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 run_mlm.py --train_file /om2/user/wongn/child-directed-listening/finetune_cut_0.25/child/Violet/train.txt --validation_file /om2/user/wongn/child-directed-listening/finetune_cut_0.25/child/Violet/val.txt --cache_dir ~/.cache/$SLURM_JOB_ID --output_dir /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/models/child/Violet/with_tags --do_eval  --do_train  --eval_steps 10 --evaluation_strategy steps --learning_rate 0.0001 --load_best_model_at_end  --logging_steps 10 --logging_strategy steps --metric_for_best_model eval_loss --model_name_or_path bert-base-uncased --num_train_epochs 10 --overwrite_output_dir  --per_device_eval_batch_size 8 --per_device_train_batch_size 8 --save_steps 10 --save_strategy steps --save_total_limit 1; singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu  python3 run_beta_search.py --split child --dataset Violet --context_width 0 --use_tags True --model_type childes
