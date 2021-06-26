#!/bin/bash 

# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

#SBATCH -N 1                         #  one node
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=high-capacity   #  high-capacity GPU
#module load openmind/singularity/3.2.0                     # load a singularity module
#singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu

python3 run_mlm.py \
	--model_name_or_path bert-base-uncased \
	--do_train \
	--do_eval \
	--output_dir /om2/user/wongn/childes_cont/meylan_model_output/with_tags_ \
	--dataset_name w-nicole/childes_data_with_tags_ \
	--overwrite_output_dir
