
import os
from os.path import join, exists
from utils import load_splits, load_models
from utils_model_sampling import beta_utils

def optimize_beta(model, uses_context, uses_tags, split_name, dataset_name, data_dir, grid_search = False):
    
    """
    For now, specify the model separately from the split_name/dataset_name.
    The reason for this is that there are two versions of the dataset (text-based and huggingface based) so this is to avoid confusion for now.
    """
     
    this_folder = split_gen.get_split_folder(split_name, dataset_name, data_dir)
    
    success_utts_sample = beta_utils.sample_successes(split_name, dataset_name, data_dir)
    beta_sample = beta_utils.get_beta_search_values(grid = grid_search)
    
    # Load the success utts/yyy utts information
    data_dict = load_splits.load_eval_data_all(split_name, dataset_name, data_dir)
    
    # Note that this will later need to retrain model to be on the same train/val split as the dataset.
    # Or discard the current split over the all/all.
    print('Need to change all/all models to be consistent dataset train/val split -- either huggingface one or new text files.')
    
    # initial_vocab determines the softmax mask used by BERT, so it's probably more conceptually correct
    # when comparing metrics
    # to leave it as the same over all of the splits,
    # especially since it was used over Providence in the original evaluation.
    
    initial_vocab, cmu_in_initial_vocab = load_models.get_initial_vocab_info()
    
    # Calculated over all of CHILDES (data pool for all/all split).
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