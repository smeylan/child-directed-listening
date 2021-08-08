#!/bin/bash

# NOTE : Quote it else use array to avoid problems #
# All text below in this
# 7/2/21: https://www.cyberciti.biz/faq/bash-loop-over-file/

mkdir -p /om2/user/wongn/child-directed-listening/experiments/scripts_train_eval_loss_specify/models/all/all/

FILES="./scripts_train_eval_loss_specify_linebyline/*"
for f in $FILES
do
  echo "Processing $f file..."
  sbatch $f
  cat "$f"
done