import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle5 as pickle

from src.utils import utils_child, load_splits
from mpl_toolkits.axes_grid1 import make_axes_locatable


def organize_auc_scores_as_grid(auc_df):

    '''
        Arrange AUC scores into a k * k  matrix, where k is the number of children with fine-tuned models

        Args:
        auc_df: longform dataframe of AUC scores

        Return:
        k * k matrix reflecting scores when applying each child-specific prior to the test set of each child in the datset

    '''
    
    names = load_splits.get_child_names()
    
    score_arr_list = []
    
    for data_child in names:
        
        this_list = []
        
        for prior_child in names:
            auc_result = set(auc_df[auc_df.cross_type == get_cross_type(data_child, prior_child)].auc)
            assert len(auc_result) == 1, "Multiple AUC found for a given cross type"
            this_list.append(list(auc_result)[0])
            
        score_arr_list.append(this_list)
    
    score_arr = np.array(score_arr_list)
    return score_arr
    
    
def get_success_scores(is_mean, which_key, likelihood_type, vmin, vmax, cmap):

    '''
        Top level function for comparing prior scores across 

        Args:
        is_mean: should this calculate mean or standard deviation?
        which_key: should this look at "prior_probability" or "posterior_probability"? 
        likelihood_type: should this evaluate scores from using a "wfst" or "levdist" likelihood function
        
        Return:
        A figure with the crossed scores 
    '''
    
    stat_type = 'Average' if is_mean else 'Standard Deviation'
    metric_type = f'{"Prior Surprisal" if "prior" in which_key else "Posterior Surprisal"}'
    threshold = 5 if "prior" in which_key else 1.2
    
    this_score_df, this_score_arr, test_dataset_name_list, training_dataset_name_list = organize_scores(is_mean = is_mean, which_key = which_key, likelihood_type = likelihood_type)
    this_title = f'{stat_type} {metric_type}' 
    
    this_figure = get_heatmap(this_title, this_score_arr, test_dataset_name_list, training_dataset_name_list, threshold, vmin, vmax, cmap)
    
    return this_figure
    
def get_heatmap(title, score_arr, test_dataset_name_list, training_dataset_name_list, threshold, vmin, vmax, cmap):

    '''
        Plot a heatmap using the k*k matrix of scores when applying each child-specific prior to the test set of each child in the datset

        Args:
        title: title to give the figure
        score_arr: k*k matrix of scores where k is the number of children with child-specific priors 

        Return
        A matplotlib figure

    '''

    training_dataset_name_list[-2] = 'CHILDES'
    training_dataset_name_list[-1] = 'Switchboard'


    figure = plt.figure()
    
    num_y_ticks = len(test_dataset_name_list)
    num_x_ticks = len(training_dataset_name_list)


    # For text annotations and color bar
    # 6/2 : https://www.pythonprogramming.in/heatmap-with-intermediate-color-text-annotations.html

    fig, ax = plt.subplots(figsize=(15, 15))

    plt.title(title)
    if vmin is not None or vmax is not None:
        im = ax.imshow(score_arr, vmin=vmin, vmax=vmax, cmap = "YlGnBu")
    else:
        im = ax.imshow(score_arr, cmap = cmap)    
    
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=2.5, pack_start = True)    
    plt.colorbar(im, fraction=0.046, pad=0.04, shrink=.75, cax=cax, orientation = 'horizontal')
    plt.axvline(x=5.5, color='black', linestyle='--')

    textcolors = ["k", "w"] 

    #6/2 hide ticks: https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/
    #6/2 rotation: https://www.delftstack.com/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/
    #6/2 xtick text: https://www.mathworks.com/help/matlab/creating_plots/change-tick-marks-and-tick-labels-of-graph-1.html

    plt.ylabel('Test Data From Child')
    plt.xlabel('Prior Fine-Tuned On Data From Child/Dataset')

    plt.xticks(range(num_x_ticks), training_dataset_name_list, rotation = 45)
    plt.yticks(range(num_y_ticks), test_dataset_name_list)

    for i in range(len(test_dataset_name_list)):
        for j in range(len(training_dataset_name_list)):
            this_val = round(score_arr[i][j].item(), 3)
            ax.text(j, i, this_val, ha="center", va="center", color=textcolors[this_val > threshold])

    # End taken code
    fig.add_axes(cax)


    return figure


def get_cross_type(data_child, prior_child):
    '''
        Get a model string reflecting the data child and the prior child in a conventional format 
        
        Args:
        data_child: name of the child whose data will be tested
        prior_child: named of the child whose prior will be used

        Return:
        A string with the model name
        

    '''
    return f'data-{data_child}+prior_child-{prior_child}'


def get_cross_augmented_scores(data_child, prior_child, model_type):

    '''
        Load individual score from using a specific child's fine-tuned prior on the data associated with another child

        Args:
        data_child: name of the child whose data will be tested
        prior_child: named of the child whose prior will be used

        Return:
        A pandas dataframe of scores

    '''
    
    score_path = utils_child.get_cross_path(data_child, prior_child, model_type)
    try:
        raw_scores = pd.read_pickle(score_path)
    except:
        with open(score_path, "rb") as fh:
            data = pickle.load(fh)
        path_to_protocol4 = score_path.replace('.pkl','.pkl4')
        data.to_pickle(path_to_protocol4)

        raw_scores = pd.read_pickle(path_to_protocol4)

    raw_scores['cross_type'] = get_cross_type(data_child, prior_child)
    raw_scores['data_child'] = data_child
    raw_scores['prior_child'] = prior_child
    
    return raw_scores
                   
                   
def load_all_scores():

    '''
        Load all scores for the cross-child fine-tuning analysis

        Args: None

        Return:
        A pandas dataframe with scores from all scores


    '''
    
    name_list = load_splits.get_child_names()
    all_scores = pd.concat(
        [
            get_cross_augmented_scores(data_child, prior_child, model_type)
            for data_child in name_list
            for prior_child in name_list
        ]
    )    
    return all_scores
    
    
def process_score_results(data_child, prior_child, model_type, which_key, likelihood_type, is_mean = True):

    '''
        Compute a mean or standard deviation over a column (eg prior or posterior probability) for a particular combination of child prior and child data

        Args:
        data_child: name of the child whose data will be tested
        prior_child: named of the child whose prior will be used
        which_key: should this look at "prior_probability" or "posterior_probability"? 
        likelihood_type: should this evaluate scores from using a "wfst" or "levdist" likelihood function
        is_mean: should this calculate mean or standard deviation?

        Return: averate or standard deviation of a score for a specific prior_child, data_child combination

    '''
    
    assert which_key in {'posterior_probability', 'prior_probability'}
    
    score_path = utils_child.get_cross_path(data_child, prior_child, model_type)

    try:
        scores = pd.read_pickle(score_path)
    except: 
        with open(score_path, "rb") as fh:
            data = pickle.load(fh)
        path_to_protocol4 = score_path.replace('.pkl','.pkl4')
        data.to_pickle(path_to_protocol4)

        scores = pd.read_pickle(path_to_protocol4)
    
    scores = scores.loc[(scores.set == 'success') & (scores.likelihood_type == likelihood_type)]    
    
    # Match the R analyses "sem" function.
    stdev_match_r = lambda s : np.std(s, ddof = 1)
    
    agg_func = np.mean if is_mean else stdev_match_r
    
    stats = agg_func(-np.log2(scores[which_key]))
    
    return stats

    
def organize_scores(is_mean, which_key, likelihood_type):
    
    '''
        Organize posterior or prior probabilities (as surprisals) by crossing child-specific priors and child-specific datasets

        is_mean: should this calculate mean or standard deviation?
        which_key: should this look at "prior_probability" or "posterior_probability"? 
        likelihood_type: should this evaluate scores from using a "wfst" or "levdist" likelihood function

        Returns: 
        a longform dataframe with all scores, and a k*k matrix of scores where k is the number of children

    '''

    results = defaultdict(list)

    # name_list = load_splits.get_child_names() 
    # models = ['childes' for x in range(len(load_splits.get_child_names()))] + ['childes','switchboard']    

    # for data_name in name_list:
    #     results[data_name] = [process_score_results(data_name, name_list[i], models[i], which_key, likelihood_type,  is_mean = is_mean) for i in range(len(name_list))]


    test_dataset_name_list = load_splits.get_child_names()
    training_dataset_name_list = test_dataset_name_list + ['all', 'all'] 
    model_type_list = ['childes' for x in range(len(load_splits.get_child_names()))] + ['childes','switchboard']    

    for test_dataset_name in test_dataset_name_list:
        results[test_dataset_name] = [process_score_results(test_dataset_name, training_dataset_name_list[i], model_type_list[i], which_key, likelihood_type,  is_mean = is_mean) for i in range(len(training_dataset_name_list))]

    
    results['Prior child name'] = training_dataset_name_list
    
    scores_arr = np.stack([np.array(results[name]) for name in test_dataset_name_list], axis = 0) 
    
    results_df = pd.DataFrame.from_records(results)
    
    return results_df, scores_arr, test_dataset_name_list, training_dataset_name_list