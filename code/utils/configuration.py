import os
from os.path import join, exists
import json

class Config:
    def __init__(self):

        '''
        Reads the following parameters from the JSON

        slurm_user : user on the SLURM cluster, e.g. OpenMind, formerly OpenMind

        for_reproducible: ??? {0,1} Configure to true for generating data for a reproducibility check

        slurm_root_dir : ??? can't find with grep

        local_root_dir : ???? can't find with grep

        n_beta : number of success samples used to evaluate scores of Levenshtein distance scaling parameter
        
        n_lambda :  number of success samples used to evaluate scores of the WFST path length parameter       

        n_across_time: Note this is the base pool sample, not necessarily the sample size used.

        subsamples : ??? Providence - splits
        
        val_ratio : ??? "Proportion of CHILDES to use for Validation" .2", ???
    
        child_val_eval_num : 3, ???

        childes_model_args : names of childes model subtypes to train ([["all", "all"], ["age", "young"],["age", "old"]])

    
        regenerate : specifies if CHILDES data is regenerated in Providence - Retrieve data.ipynb

        dev_mode: ??? "Whether or not to truncate number of samples, etc. (for development purposes)" Used in config_train.py, gen_training_scripts, gen_sample scripts. Seems to decrease the scale of training and test 

        subsample_mode: ??? n_beta, n_across_time for faster iteration. How does this relate

        n_iter_sample : number of samples in a normal (non-dev) run 

        n_dev_sample : number of samples in a dev run 

        dist_type : ??? type of the distance function to use for the likeklioof function, now we compute both sp we can drop it

        eval_phase: {'val', 'eval'} -- ??? what to compute the scores on. Switch to eval at the end?

        exp_determiner: Name of the model run in which to place all results in experiments/

        training_version_name : Name of the model run, in case you want to use different trained models for scoring. This allows an experiment folder to have scores but no trained models. Doesn't appear to be in use

        child_context_width: ??? ow many different context widths to use
    
        verbose: {0,1}, useful for debugging or data generation.

        age_split : age in months to distinguish old vs. young children (30)

        context_list : list of context widths to test ([0, 20])

        beta_low : lowest value of beta to test (2.5)
        
        beta_high : highest value of beta to test (4.5)
    
        beta_num_values : number of values to test between the low and the high value of beta(20)

        fail_on_beta_edge : should the code fail if the best value is on the edge of the range of betas tested? (1)

        lambda_low : lowest value of lambda to test (0)

        lambda_high : highest value of lambda to test (2)

        lambda_num_values : number of values to test between the low and the high value of lambda(20)   

        fail_on_lambda_edge : should the code fail if the best value is on the edge of the range of lambdas tested? (1)

        fst_path:  path to the fst txt file used for the WFST ("fst/chi-1.txt")

        fst_sym_path: path to the phones file used for the WFST ("fst/chi_phones.sym")

        fst_cache_path: path to where pairwise WFST path lengths between data and all vocab will be stored ("unigram_fst_cache")

        
        '''

        # read in the environment variables        
        self.json_path = 'config.json'
        
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

        if not hasattr(self, 'local_root_dir'):
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
        elif self.subsample_mode:
            self.n_subsample = self.n_iter_sample
        else:
            self.n_subsample = self.n_beta
        
        self.n_used_score_subsample = self.n_subsample # Used for loading the scores in the analyses.

        self.exp_dir = join(join(self.local_root_dir, 'experiments'), self.exp_determiner)

        self.model_dir = join(self.exp_dir, 'models')

        self.scores_dir = join(self.exp_dir, join('scores', join(f'n={self.n_used_score_subsample if (self.subsample_mode or self.dev_mode) else self.n_beta}', self.eval_phase))) # Beta, across time, cross-child scoring.

        self.model_analyses_dir = join(self.exp_dir, 'model_analyses')


        #########################
        ## TRAINING ARGUMENTS ##
        #########################

        self.general_training_args = {            
            'model_name_or_path' : 'bert-base-uncased',            
            'num_train_epochs' : 3,
            'learning_rate' : 5e-5,            
            'eval_steps' : 500 if not self.dev_mode else 1,
            'logging_steps' : 500 if not self.dev_mode else 1,
            'save_steps' : 500 if not self.dev_mode else 1,
        }


        ###########################################
        #### CHILD-SPECIFIC TRAINING ARGUMENTS ####
        ###########################################


        self.version_name = self.training_version_name # Separate from exp determiner, because you may want to generate separate training files than scoring on Chompsky
        self.exp_dir = join(join(self.local_root_dir, 'experiments'), self.version_name)
        self.model_dir = join(self.exp_dir, 'models')


        self.child_args = {
            
            'model_name_or_path' : load_models.get_model_path('all', 'all', True),
            
            'num_train_epochs' : 10,
            'learning_rate' : 5e-5, # Unsure, need to check Alex convergence etc.

            'eval_steps' : 100 if not self.dev_mode else 10,
            'logging_steps' : 100 if not self.dev_mode else 10,
            'save_steps' : 100 if not self.dev_mode else 10,
            
        }

        base_args = {
    
            # Boolean arguments: basically pass in the argument --do_train, which signifies True
            'do_train' : '', 
            'do_eval': '',
            'load_best_model_at_end' : '',
            'overwrite_output_dir' : '',
            
            'metric_for_best_model' : 'eval_loss',
            
            'evaluation_strategy' : 'steps',
            'save_strategy' : 'steps',
            'logging_strategy' : 'steps',
            'save_total_limit' : 1,
            
            # Always overwrite by default. Note child arguments load from a model path, not a trainer checkpoint.
            
            'per_device_train_batch_size' : 8, # Maximal for linebyline = False, 9 GB GPU.,
            'per_device_eval_batch_size'  : 8 # Maximal for linebyline = False, 9 GB GPU.,
            
        }

        self.child_args.update(base_args)
        self.general_training_args.update(base_args)

    def check(self):
        assert self.n_beta == self.n_across_time, "The codebase generally assumes this for convenience."

    def make_folders(self, paths):
        for p in paths:
            if not exists(p):
                os.makedirs(p)