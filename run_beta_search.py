
import os
from os.path import join, exists
from utils import load_splits, load_models, split_gen, parsers
from utils_model_sampling import beta_utils, sample_across_models
from utils_child import child_models

import config
import pandas as pd

import matplotlib.pyplot as plt
 
import numpy as np

import argparse

def optimize_beta(split_name, dataset_name, model_dict, model_type):
    
    """
    For now, specify the model separately from the split_name/dataset_name.
    The reason for this is that there are two versions of the dataset (text-based and huggingface based) so this is to avoid confusion for now.
    
    model_dict = the dictionary entry as specified in yyy
    """
 
    beta_sample = beta_utils.get_beta_search_values()
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    
    this_exp_path = beta_utils.load_beta_folder(split_name, dataset_name, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type)
    
    if not exists(this_exp_path):
        os.makedirs(this_exp_path)
    
    # Calculated over all of CHILDES (data pool for all/all split).
    # Internally uses GPU if available.
    # speaker tags handled internally in the transformers bert completions file.
    
    # 7/15/21: Orthogonally changing this to branch on child split -- was not present in original rep. code
    if split_name != 'child':
        success_utts_sample = load_splits.load_sample_successes('beta', split_name, dataset_name)
        yyy_utts_sample = load_splits.load_sample_yyy('beta', split_name, dataset_name)
    else:
        # Optimize on the entirety of the child set.
        # TODO: implement this
        
        print('You need to implement this!')
        
        data_dict = load_splits.load_pvd_data(split_name, dataset_name, config.eval_phase)
        success_utts_sample = data_dict['success_utts']
        yyy_utts_sample = data_dict['yyy_utts']
        
    this_raw_beta_results = sample_across_models.sample_across_models(success_utts_sample,
                                                                      yyy_utts_sample,
                                                                      model_dict,
                                                                      beta_sample)
    
    this_beta_results_surp = this_raw_beta_results.groupby(['beta_value']).posterior_surprisal.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    
    # Log the beta results
    beta_results_path = join(this_exp_path, f'beta_search_results_{config.n_beta}.csv')
    
    this_raw_beta_results.to_csv(join(this_exp_path, f'beta_search_raw_results_{config.n_beta}.csv')) # May not need to save this.
    this_beta_results_surp.to_csv(beta_results_path)
    
    print("Writing beta results to", {beta_results_path})
    
    plot_beta_optimization(this_exp_path, beta_sample, this_beta_results_surp['posterior_surprisal'], split_name, dataset_name)
    
    return this_raw_beta_results, this_beta_results_surp
    
def plot_beta_optimization(fig_path_dir, betas, beta_surprisals, split, dataset):
    
    plt.title(f'Beta optimization for Split: {split}, Dataset: {dataset}')
    plt.xlabel('Beta value')
    plt.ylabel('Posterior surprisal')
    plt.scatter(betas, beta_surprisals)
    
    fig_path = join(fig_path_dir, f'beta_optimization_{config.n_beta}.png')
    plt.savefig(fname = fig_path)
    
    print(f'Writing optimization plot to: {fig_path}')
    return fig_path
    
if __name__ == '__main__':
    
    parser = parsers.split_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # end cite
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_model_args = vars(raw_args)
    
    query_model_str = load_models.get_model_id(
        split_name = this_model_args['split'],
        dataset_name = this_model_args['dataset'],
        with_tags =  this_model_args['use_tags'],
        context_width = this_model_args['context_width'],
        model_type = this_model_args['model_type']
    )
    
    print(this_model_args)
   
    if this_model_args['split'] != 'child':
        this_model_dict = load_models.get_specific_model_dict(query_model_str)
    else:
        # Note that model args should already be matched to the parse arguments for child scripts,
        # because they are auto-generated.
        # But assert anyway to prevent manual misuse
        
        this_model_dict = child_models.get_child_model_dict(this_model_args['dataset'])
        
        assert this_model_dict['kwargs']['use_speaker_labels'] == this_model_args['use_tags']
        assert this_model_dict['kwargs']['context_width_in_utts'] == this_model_args['context_width']
        
    raw_results, beta_results = optimize_beta(this_model_args['split'], this_model_args['dataset'], this_model_dict, this_model_args['model_type'])

    print(f'Computations complete for: {query_model_str}')
    
    