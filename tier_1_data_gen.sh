# Remove any directories with cached results
rm -rf output

# Remove any previously processed notebooks
rm *.nbconvert.ipynb

# Run notebooks that prepare the data and generate the data splits
jupyter nbconvert --execute 'src/tier_1/1 - Create Vocabulary and Pronunciations.ipynb' --to notebook
jupyter nbconvert --execute 'src/tier_1/2 - Providence - Retrieve Data.ipynb' --to notebook
jupyter nbconvert --execute 'src/tier_1/3 - Providence - Splits.ipynb' --to notebook
jupyter nbconvert --execute 'src/tier_1/4 - Get Non-Providence CHILDES Finetuning Data.ipynb' --to notebook
