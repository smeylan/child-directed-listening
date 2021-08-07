# #!/bin/bash

# NOTE : Quote it else use array to avoid problems #
# All text below in this
# 7/2/21: https://www.cyberciti.biz/faq/bash-loop-over-file/

FILES="./scripts_train_lr_0.001_epochs_5/*"
for f in $FILES
do
  echo "Processing $f file..."
  sbatch $f
  cat "$f"
done