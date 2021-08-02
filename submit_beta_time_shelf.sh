# #!/bin/bash

# For the command text
# 6/24/21: https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-use-Singularity-container%3F
# and https://github.mit.edu/MGHPCC/OpenMind/issues/3392
# including the bash line at the top, and all but the python3 commands

#SBATCH -N 1
#SBATCH -p cpl
#SBATCH -t 00:10:00
#SBATCH --mem=50M

module load openmind/singularity/3.2.0

singularity exec --nv -B /om,/om2/user/wongn /om2/user/wongn/vagrant/trans-pytorch-gpu python3 gen_sample_scripts.py

# NOTE : Quote it else use array to avoid problems #
# All of the text in this, including the header and comment above, are from
# 7/2/21: https://www.cyberciti.biz/faq/bash-loop-over-file/
FILES="./scripts_beta_time/shelf/*"
for f in $FILES
do
  echo "Processing $f file..."
  sbatch $f
  cat "$f"
done