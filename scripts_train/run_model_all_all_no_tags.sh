#!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH --gres=gpu:1
#SBATCH -t 7:00:00
#SBATCH --mem=9G
#SBATCH --constraint=high-capacity

module load openmind/singularity/3.2.0
#!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH --gres=gpu:1
#SBATCH -t 7:00:00
#SBATCH --mem=9G
#SBATCH --constraint=high-capacity

module load openmind/singularity/3.2.0
singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu     python3 run_mlm.py             --model_name_or_path bert-base-uncased             --do_train             --do_eval             --output_dir /om2/user/wongn/childes_run/child-directed-listening/models/all/all/no_tags            --train_file /om2/user/wongn/childes_run/child-directed-listening/data/new_splits/all/all/train_no_tags.txt             --validation_file /om2/user/wongn/childes_run/child-directed-listening/data/new_splits/all/all/val_no_tags.txt             --overwrite_output_dir