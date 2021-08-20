#!/bin/bash

chmod u+x tier_2a_non_child_train_shelf_scores.sh
chmod u+x tier_2b_finetune_scores_child_train.sh

./tier_2a_non_child_train_shelf_scores.sh
./tier_2b_finetune_scores_child_train.sh

# Then the following on Chompsky:
# tmux attach-session -t experiments
# rsync -a --progress wongn@openmind.mit.edu:~/child_repo_split/experiments ./experiments