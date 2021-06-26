
import os
from os.path import join, exists
from utils import load_splits
from utils_model_sampling import beta_utils

def optimize_beta(split_name, dataset_name, data_dir, grid_search = False):
    
    # You should also load the right model.
     
    this_folder = split_gen.get_split_folder(split_name, dataset_name, data_dir)
    
    success_utts_sample = beta_utils.sample_successes(split_name, dataset_name, data_dir)
    beta_sample = beta_utils.get_beta_search_values(grid = grid_search)
    
    # Load the success utts/yyy utts information
    data_dict = load_splits.load_eval_data_all(split_name, dataset_name, data_dir)
    
    # Load the right model?
    
    
    this_raw_beta_results = transfomers_bert_completions.sample_across_models(success_utts_sample, data_dict['success_utts'], data_dict['yyy_utts'], all_tokens_phono['yyy_utts'], models[0:1], initial_vocab, cmu_in_initial_vocab, beta_values =
                     test_beta_vals) # Notice the use of models here.
    
    this_beta_results_surp = this_raw_beta_results.groupby(['beta_value']).posterior_surprisal.agg(lambda x: np.mean(-1 * np.log(x))
).reset_index()
    
    # Log the beta results
    this_raw_beta_results.to_csv(join(this_folder, 'beta_search_raw_results.csv')) # May not need to save this.
    this_beta_results_surp.to_csv(join(this_folder, 'beta_search_results.csv'))
    
    plot_beta_optimization(beta_sample, this_beta_results_surp)
    
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
    
    sample_data(data_dir)
    
    pass