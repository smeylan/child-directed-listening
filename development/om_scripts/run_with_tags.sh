#!/bin/bash
#from 6/18 https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F

python3 run_mlm_with_tags.py \
 --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --dataset_name w-nicole/childes_data_with_tags \
    --output_dir /om2/user/wongn/childes_cont/meylan_model_output/with_tags \
    --overwrite_output_dir
