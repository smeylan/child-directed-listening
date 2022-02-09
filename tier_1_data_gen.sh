# Remove any directories with cached results
rm -rf finetune
rm -rf prov
rm -rf prov_csv
rm -rf figures
rm -rf phon
rm -rf fst


# Re-create the figures directory
mkdir -p figures

# Remove any previously processed notebooks
rm *.nbconvert.ipynb

# Run notebooks that prepare the data and generate the data splits
jupyter nbconvert --execute 'code/tier_1/1 - Create Vocabulary and Pronunciations.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_1/2 - Generalized Phonological Comparison' --to notebook
jupyter nbconvert --execute 'code/tier_1/3 - Providence - Retrieve Data.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_1/4 - Providence - Splits.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_1/5 - Get Non-Providence CHILDES Finetuning Data.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_1/6 - Output Citation vs. Observed Phonology Pairs.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_1/7 - Data Splitting Checks.ipynb' --to notebook
bash 'code/tier_1/8 - Train FST.sh'


# Then, rsync the resulting files to the SLURM cluster
rsync -az --progress ./finetune ./prov ./prov_csv ./csv ./phon ./fst ${SLURM_USERNAME}:${CDL_SLURM_ROOT}