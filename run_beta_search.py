
import os
from os.path import join, exists
from utils import load_splits, load_models
from utils_model_sampling import beta_utils

def optimize_beta(model_dict, uses_tags, uses_context, split_name, dataset_name, data_dir, grid_search = False):
    
    """
    For now, specify the model separately from the split_name/dataset_name.
    The reason for this is that there are two versions of the dataset (text-based and huggingface based) so this is to avoid confusion for now.
    
    model_dict = the dictionary entry as specified in yyy
    """
     
    model = model_dict['kwargs']['modelLM']
    this_folder = split_gen.get_split_folder(split_name, dataset_name, data_dir)
    
    success_utts_sample = load_splits.sample_successes('beta', split_name, dataset_name, data_dir)
    beta_sample = beta_utils.get_beta_search_values(grid = grid_search)
    
    # Load the success utts/yyy utts information
    data_dict = load_splits.load_eval_data_all(split_name, dataset_name, data_dir)
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    this_exp_path = join(this_folder, model['title'].replace(' ', '_'))
    
    if not exists(this_exp_path):
        os.makedirs(this_exp_path)
    
    # Calculated over all of CHILDES (data pool for all/all split).
    # Internally uses GPU if available.
    
    # Need to remove speaker tags from?
    this_raw_beta_results = transfomers_bert_completions.sample_across_models(success_utts_sample,
                                                                              data_dict['success_utts'],
                                                                              data_dict['yyy_utts'],
                                                                              model,
                                                                              initial_vocab,
                                                                              cmu_in_initial_vocab,
                                                                              beta_values = beta_sample)
    
    this_beta_results_surp = this_raw_beta_results.groupby(['beta_value']).posterior_surprisal.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    
    # Log the beta results
    beta_results_path = join(this_exp_path, 'beta_search_results.csv')
    
    this_raw_beta_results.to_csv(join(this_exp_path, 'beta_search_raw_results.csv')) # May not need to save this.
    this_beta_results_surp.to_csv(beta_results_path)
    
    print("Writing beta results to", {beta_results_path})
    
    plot_beta_optimization(beta_sample, this_beta_results_surp, this_exp_path)
    
    return this_raw_beta_results, this_beta_results_surp
    
def plot_beta_optimization(split, dataset, data_dir, betas, beta_surprisals):
    
    plt.title(f'Beta optimization for Split: {split}, Dataset: {dataset}')
    plt.xlabel('Beta value')
    plt.ylabel('Posterior surprisal')
    plt.plot(betas, beta_surprisals)
    
    fig_path = join(data_dir, 'beta_optimization.png')
    plt.savefig(fname = fig_path)
    
    print(f'Writing optimization plot to: {fig_path}')
    return fig_path
    
if __name__ == '__main__':
    
    regenerate = False
    data_dir_base = 'eval/new_splits'
    # This is where the evaluation data is found.
    
    # Be careful of this! Because the 'all' split currently is not the same for the two models.
    # For reproducibility purposes, redirect loading to all_old.
    
    # Consider using argparse + parallel jobs instead.
    
    which_args = [('all', 'all'), ('age', 'old'), ('age', 'young')]
    
    this_split = 'all_old'
    this_dataset_name = 'all_old'
    use_tags = False
    use_context = False
    
    tags_str = f"{'with' if use_tags else 'no'}_tags"
    context_str = f"{20 if use_context else 0}_context" 
        
    this_utts_sample = load_splits.sample_successes('beta', this_split, this_dataset_name, data_dir_base, n = 5, regenerate = False)
    this_beta_values_sample = beta_utils.get_beta_search_values(num_values = 2) # For now, because you are just getting code to run.
    
    # This is currently only designed for querying BERT models -- generalize this later.
    query_model_str = f'{this_split}/{this_dataset_name}/{tags_str}/{context_str}'
    all_models = load_models.get_model_dict('./')[query_model_str] # This is probably going to be slow, optimize later
    
    raw_results, beta_results = optimize_beta(this_model_dict, use_tags, use_context,
                                              this_split, this_dataset_name, data_dir_base)
    
    print(f'Computations complete for: {query_model_str}')