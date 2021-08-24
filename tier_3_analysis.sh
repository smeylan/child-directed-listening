
rm 'Examples for Succcess and Failure Table.nbconvert.ipynb'
rm 'Models across time analyses.nbconvert.ipynb'
rm 'Child evaluations.nbconvert.ipynb'
rm 'Prevalence analyses.nbconvert.ipynb'

jupyter nbconvert --execute 'Examples for Succcess and Failure Table' --to notebook
jupyter nbconvert --execute 'Models across time analyses' --to notebook
jupyter nbconvert --execute 'Child evaluations' --to notebook
jupyter nbconvert --execute 'Prevalence analyses' --to notebook

