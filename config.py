# 7/3/21: For programming style and best practices (the use of a config.py, not code in particular)
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

import os
from os.path import join, exists

def make_folders(paths):
    for p in paths:
        if not exists(p):
            os.makedirs(p)

om_user = 'wongn'

for_reproducible = False # Configure to true for generating data for a reproducibility check
reproducibility_modifier = '_for_rep' if for_reproducible else ''

root_dir = os.getcwd()

# Script generation is moved to Chompsky
om_root_dir = '/om2/user/wongn/child-directed-listening' if 'chompsky' in root_dir else root_dir

### --- Data splitting arguments

SEED = 0 

n_beta = 5000
n_across_time = 5000 # Note this is the base pool sample, not necessarily the sample size used.

assert n_beta == n_across_time, "The codebase generally assumes this for convenience."

subsamples = [2, 500, 1000] # 2, 500 are for development purposes

val_ratio = 0.2 # For the CHILDES split.

child_val_eval_num = 3

finetune_dir_name = f'finetune{reproducibility_modifier}'
finetune_dir = join(root_dir, finetune_dir_name)

# Beta and across time evaluations
prov_dir = join(root_dir, f'prov{reproducibility_modifier}') # Location of data for evaluations (in yyy)

prov_csv_dir = join(root_dir, 'prov_csv')
cmu_path = join(root_dir, 'phon/cmu_in_childes.pkl') # The location of the cmu dictionary

# Non-child arguments only.
childes_model_args = [('all', 'all'), ('age', 'young'), ('age', 'old')]

make_folders([finetune_dir, prov_dir, prov_csv_dir])

### --- End data splitting arguments

regenerate = True # Whether to regenerate data or long-running computations
dev_mode = True # Whether or not to truncate number of samples, etc. (for development purposes)

# These will override n_beta, n_across_time for faster iteration
subsample_mode = True
n_iter_sample = 500 # Amount used for faster analysis iteration.
n_dev_sample = 2 # Amount used for development/debugging.

# Note: in the future, always use subsample argument unless revert to 5000

if dev_mode:
    n_subsample = n_dev_sample
elif subsample_mode:
    n_subsample = n_iter_sample
else:
    n_subsample = n_beta
    
n_used_score_subsample = n_subsample # Used for loading the scores in the analyses.


verbose = True # True for debugging or data generation.

dist_type = 'levdist' # What distance scoring function to use. Need to enforce this throughout the code later.
eval_phase = 'val' # {'val', 'eval'} -- what to compute the scores in 


# Which experimental set of models to use.
# Datetime is determined by the model training script generation.

exp_determiner = 'no_versioning'
exp_dir = join(join(root_dir, 'experiments'), exp_determiner)

model_dir = join(exp_dir, 'models')
scores_dir = join(exp_dir, join('scores', join(f'n={n_used_score_subsample if (subsample_mode or dev_mode) else n_beta}', eval_phase))) # Beta, across time, cross-child scoring.
    
make_folders([model_dir, scores_dir])

model_analyses_dir = join(exp_dir, 'model_analyses') # For intermediate model visualizations

child_context_width = 0
context_list = [0, 20] # How many different context widths to use
age_split = 30 # Split young <= {age_split} months. (affects data generation only, not sample loading after the split)

beta_low = 2.5
beta_high = 3.5
num_values = 10

