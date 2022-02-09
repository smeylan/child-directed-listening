rm 'Examples for Succcess and Failure Table.nbconvert.ipynb'
rm 'Models across time analyses.nbconvert.ipynb'
rm 'Child evaluations.nbconvert.ipynb'
rm 'Prevalence analyses.nbconvert.ipynb'

jupyter nbconvert --execute 'code/tier_3/1 - Examples for Success and Failure Table.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_3/2 - Models Across Time Analyses.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_3/3 - Child Evaluations.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_3/4 - Prevalence Analyses.ipynb' --to notebook

