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
#SBATCH --output=/om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/scores/n=500/val/child/Alex/%j_training_beta_tags=False.out
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/scores/n=500/val/child/Alex

module load openmind/singularity/3.2.0
rm -r /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/models/child/Alex
mkdir ~/.cache/$SLURM_JOB_ID
# 7/13/21: https://stackoverflow.com/questions/19960332/use-slurm-job-id for variable name of job ID
 python3 run_mlm.py --train_file /home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening/finetune/child/Alex/train_no_tags.txt --validation_file /home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening/finetune/child/Alex/val_no_tags.txt --cache_dir ~/.cache/$SLURM_JOB_ID --output_dir /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/models/child/Alex/no_tags --do_eval  --do_train  --eval_steps 10 --evaluation_strategy steps --learning_rate 5e-05 --load_best_model_at_end  --logging_steps 10 --logging_strategy steps --metric_for_best_model eval_loss --model_name_or_path bert-base-uncased --num_train_epochs 10 --overwrite_output_dir  --per_device_eval_batch_size 8 --per_device_train_batch_size 8 --save_steps 10 --save_strategy steps --save_total_limit 1; singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu  python3 run_beta_search.py --split child --dataset Alex --context_width 0 --use_tags False --model_type childes
