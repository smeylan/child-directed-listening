# 7/3/21: For programming style and best practices
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

import os
from os.path import join

# Still need to refactor code to actually use this file

SEED = 0 
regenerate = False # Whether to regenerate data or long-running computations

n_beta = 5 # For debugging
n_across_time = 1 # For debugging

verbose = False # True for debugging.
val_ratio = 0.2 # For the CHILDES split.
child_val_num = 200

#n_beta = 5000 # Number of samples for beta fitting
#n_across_time = 1000 # Number of samples across time

# Note that if this is specified to "None", it really defaults to:
# beta fitting on 5000 samples
# and running models across time on 1000 samples.

root_dir = '/home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening' # Location of repository
# root_dir = '/om2/user/wongn/childes_run/child-directed-listening' # Location of repository

data_dir = join(root_dir, 'data/new_splits') # Location of data for model fitting
eval_dir = join(root_dir, 'eval/new_splits') # Location of data for evaluations (in yyy)
model_dir = join(root_dir, 'models/new_splits') # Location of model weights, etc.
exp_dir = join(root_dir, 'scores') # Where to put the sampling results (beta searching, models across time)

meylan_model_dir = join(root_dir, 'models') # Possibly temporary -- location of root for Dr. Meylan's old models

cmu_path = join(root_dir, 'phon/cmu_in_childes.csv') # The location of the cmu dictionary

context_list = [0, 20] # How many different context widths to use
age_split = 36 # Split young <= {age_split} months.

beta_low = 2.5
beta_high = 3.5
num_values = 10
grid_search = False # Whether to grid search the beta parameters or use random search.
# above: Note, this is still unused/untested.

# Which models to use in the analyses -- primarily for running the sampling-based functions.
childes_model_args = [('all', 'all'), ('age', 'young'), ('age', 'old')]

