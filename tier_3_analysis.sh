
rm *.nbconvert.ipynb

jupyter nbconvert --execute 'Examples for Succcess and Failure Table' --to notebook
jupyter nbconvert --execute 'Models across time analyses' --to notebook
jupyter nbconvert --execute 'Child evaluations' --to notebook