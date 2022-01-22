# Remove any directories with cached results
rm -rf finetune
rm -rf prov
rm -rf prov_csv
rm -rf figures

# Re-create the figures directory
mkdir -p figures

# Remove any previously processed notebooks
rm *.nbconvert.ipynb

# Run notebooks that prepare the data and generate the data splits
jupyter nbconvert --execute 'Create Vocabulary and Pronunciations.ipynb' --to notebook
jupyter nbconvert --execute 'Generalized Phonological Comparison' --to notebook
jupyter nbconvert --execute 'Providence - Retrieve data.ipynb' --to notebook
jupyter nbconvert --execute 'Providence - Splits.ipynb' --to notebook
jupyter nbconvert --execute 'Get non-Providence CHILDES finetuning data.ipynb' --to notebook
jupyter nbconvert --execute 'Output Citation vs. Observed Phonology Pairs.ipynb' --to notebook
jupyter nbconvert --execute 'data_splitting_checks.ipynb' --to notebook
train_fst.sh


# Then, rsync the resulting files to the SLURM cluster
rsync -az --progress ./finetune ./prov ./prov_csv ./csv ./phon ./fst ${SLURM_USERNAME}:${CDL_SLURM_ROOT}