
import os
from os.path import join, exists
from utils import load_splits, load_models, split_gen, parsers
from utils_model_sampling import beta_utils, sample_across_models

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
    
    # Load the success utts/yyy utts information
    data_dict = load_splits.load_eval_data_all(split_name, dataset_name)
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    
    this_exp_path = beta_utils.load_beta_folder(split_name, dataset_name, model_dict['kwargs']['use_speaker_labels'], model_dict['kwargs']['context_width_in_utts'], model_type)
    
    if not exists(this_exp_path):
        os.makedirs(this_exp_path)
    
    # Calculated over all of CHILDES (data pool for all/all split).
    # Internally uses GPU if available.
    # speaker tags handled internally in the transformers bert completions file.
    
    # Only works for now on BERT models. Need to figure out the unigram stuff afterwards.
    
    success_utts_sample = load_splits.load_sample_successes('beta', split_name, dataset_name)
    yyy_utts_sample = load_splits.load_sample_yyy('beta', split_name, dataset_name)
    
    total_sample = pd.concat([success_utts_sample, yyy_utts_sample]).utterance_id
    # Extraction of attribute is necessary to have non-empty extraction of sample
    
    this_raw_beta_results = sample_across_models.sample_across_models(total_sample,
                                                                      model_dict,
                                                                      data_dict,
                                                                      beta_sample)
    
    this_beta_results_surp = this_raw_beta_results.groupby(['beta_value']).posterior_surprisal.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    
    # Log the beta results
    beta_results_path = join(this_exp_path, 'beta_search_results.csv')
    
    this_raw_beta_results.to_csv(join(this_exp_path, 'beta_search_raw_results.csv')) # May not need to save this.
    this_beta_results_surp.to_csv(beta_results_path)
    
    print("Writing beta results to", {beta_results_path})
    
    plot_beta_optimization(this_exp_path, beta_sample, this_beta_results_surp['posterior_surprisal'], split_name, dataset_name)
    
    return this_raw_beta_results, this_beta_results_surp
    
def plot_beta_optimization(fig_path_dir, betas, beta_surprisals, split, dataset):
    
    plt.title(f'Beta optimization for Split: {split}, Dataset: {dataset}')
    plt.xlabel('Beta value')
    plt.ylabel('Posterior surprisal')
    plt.plot(betas, beta_surprisals)
    
    fig_path = join(fig_path_dir, 'beta_optimization.png')
    plt.savefig(fname = fig_path)
    
    print(f'Writing optimization plot to: {fig_path}')
    return fig_path
    
if __name__ == '__main__':
    
    parser = parsers.split_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # Not sure why known args is necessary here.
    
    # parsers.check_args(raw_args)
    
    this_model_args = vars(raw_args)
    
    query_model_str = load_models.get_model_id(
        split_name = this_model_args['split'],
        dataset_name = this_model_args['dataset'],
        with_tags =  this_model_args['use_tags'],
        context_width = this_model_args['context_width_in_utts'],
        model_type = this_model_args['model_type']
    )
   
    this_model_dict = load_models.get_model_dict()[query_model_str] # This is probably going to be slow, optimize later
    raw_results, beta_results = optimize_beta(this_model_args['split'], this_model_args['dataset'], this_model_dict, this_model_args['model_type'])
    
    print(f'Computations complete for: {query_model_str}')
    
    