# Remove files with cached results
rm -rf finetune
rm -rf prov
rm -rf prov_csv

# Remove any notebooks that have been previously run
rm *.nbconvert.ipynb

# Run notebboks that generate the data splits
jupyter nbconvert --execute 'Process CMU dictionary.ipynb' --to notebook
jupyter nbconvert --execute 'Generalized Phonological Comparison' --to notebook
jupyter nbconvert --execute 'Providence - Retrieve data.ipynb' --to notebook & jupyter nbconvert --execute 'Get non-Providence CHILDES finetuning data.ipynb' --to notebook
jupyter nbconvert --execute 'Providence - Splits.ipynb' --to notebook
jupyter nbconvert --execute 'data_splitting_checks.ipynb' --to notebook

# Then, rsync the resulting files to the SLURM cluster
rsync -az --progress ./finetune ./prov ./csv ./phon ${SLURM_USERNAME}:${CDL_SLURM_ROOT}