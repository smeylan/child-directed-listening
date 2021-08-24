import os
from os.path import join, exists
import json

class Config:
    def __init__(self):

        '''
        Reads the following parameters from the JSON
        
        slurm_user: user on the SLURM cluster, e.g. OpenMind, formerly OpenMind

        fail_on_beta_edge: if true, then fails if grid sampling shows that the best beta is at the edge of the tested interval

        for_reproducible: {0,1} Configure to true for generating data for a reproducibility check

        n_across_time: Note this is the base pool sample, not necessarily the sample size used.

        regenerate: {0,1} Whether to regenerate data or long-running computations 

        dev_mode: {0,1} Whether or not to truncate number of samples, etc. (for development purposes)

        verbose: {0,1}, useful for debugging or data generation.

        dist_type = 'levdist' what distance scoring function to use. Need to enforce this throughout the code later.

        eval_phase: {'val', 'eval'} -- what to compute the scores in 

        exp_determiner: Which experimental set of models to use.

        child_context_width: How many different context widths to use

        val_ratio: proportion of CHILDES to use for validation

        subsample_mode: n_beta, n_across_time for faster iteration

        '''
        self.json_path = os.environ['CDL_CONFIG_PATH']
        
        # read in the JSON path and set everything
        f = open(self.json_path,)
        data = json.load(f)
        
        #set keys and vals from the JSON
        for key, value in data.items():
            setattr(self, key, value)
        
        self.set_defaults()
        self.check()
        self.make_folders([self.finetune_dir, self.prov_dir, self.prov_csv_dir])
        self.make_folders([self.model_dir, self.scores_dir, self.model_analyses_dir])


    def set_defaults(self):
        # compute defaults

        if not self.local_root_dir:
            self.local_root_dir = os.getcwd()
        
        self.reproducibility_modifier = '_for_rep' if self.for_reproducible else ''
        self.finetune_dir_name = f'finetune{self.reproducibility_modifier}'
        self.finetune_dir = join(self.local_root_dir, self.finetune_dir_name)

        # Beta and across time evaluations
        self.prov_dir = join(self.local_root_dir, f'prov{self.reproducibility_modifier}') # Location of data for evaluations (in yyy)

        self.prov_csv_dir = join(self.local_root_dir, 'prov_csv')
        self.cmu_path = join(self.local_root_dir, 'phon/cmu_in_childes.pkl') # The location of the cmu dictionary
 
        if self.dev_mode:
            self.n_subsample = self.n_dev_sample
        elif subsample_mode:
            self.n_subsample = self.n_iter_sample
        else:
            self.n_subsample = self.n_beta
        
        self.n_used_score_subsample = self.n_subsample # Used for loading the scores in the analyses.

        self.exp_dir = join(join(self.local_root_dir, 'experiments'), self.exp_determiner)

        self.model_dir = join(self.exp_dir, 'models')

        self.scores_dir = join(self.exp_dir, join('scores', join(f'n={self.n_used_score_subsample if (self.subsample_mode or self.dev_mode) else self.n_beta}', self.eval_phase))) # Beta, across time, cross-child scoring.

        self.model_analyses_dir = join(self.exp_dir, 'model_analyses')

    def check(self):
        assert self.n_beta == self.n_across_time, "The codebase generally assumes this for convenience."

    def make_folders(self, paths):
        for p in paths:
            if not exists(p):
                os.makedirs(p)