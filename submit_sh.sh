# #!/bin/bash
# NOTE : Quote it else use array to avoid problems #
# All of the text in this, including the header and comment above, are from
# 7/2/21: https://www.cyberciti.biz/faq/bash-loop-over-file/
FILES="/om2/user/wongn/childes_run/train_scripts/*"
for f in $FILES
do
  echo "Processing $f file..."
  sbatch $f
  cat "$f"
done
