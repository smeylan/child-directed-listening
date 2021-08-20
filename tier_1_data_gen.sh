
rm *.nbconvert.ipynb

jupyter nbconvert --execute 'Process CMU dictionary.ipynb' --to notebook
jupyter nbconvert --execute 'Generalized Phonological Comparison' --to notebook
jupyter nbconvert --execute 'Providence - Retrieve data.ipynb' --to notebook & jupyter nbconvert --execute 'Get non-Providence CHILDES finetuning data.ipynb' --to notebook
jupyter nbconvert --execute 'Providence - Splits.ipynb' --to notebook
jupyter nbconvert --execute 'data_splitting_checks.ipynb' --to notebook

# Then, rsync everything to OM.
 
# tmux new-session -s rsync_finetune
# rsync -a --progress ./finetune wongn@openmind.mit.edu:~/child_repo_split

# tmux new-session -s rsync_prov
#rsync -a --progress ./prov wongn@openmind.mit.edu:~/child_repo_split

# tmux new-session -s rsync_csv
# rsync -a --progress ./csv wongn@openmind.mit.edu:~/child_repo_split

# tmux new-session -s rsync_phon
# rsync -a --progress ./phon wongn@openmind.mit.edu:~/child_repo_split


