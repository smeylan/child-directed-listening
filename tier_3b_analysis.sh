rm -rf 'src/tier_3/Examples for Succcess and Failure Table.nbconvert.ipynb'
rm -rf 'src/tier_3/Models across time analyses.nbconvert.ipynb'
rm -rf 'src/tier_3/Child evaluations.nbconvert.ipynb'


jupyter nbconvert --execute 'code/tier_3/1 - Examples for Success and Failure Table.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_3/2 - Models Across Time Analyses.ipynb' --to notebook
jupyter nbconvert --execute 'code/tier_3/3 - Child Evaluations.ipynb' --to notebook