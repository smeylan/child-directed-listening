
import pandas as pd
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils_child import utils_child, child_models


def organize_auc_scores_as_grid(auc_df):
    
    names = child_models.get_child_names()
    
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
    
    
def get_success_scores(is_mean, which_key):
    
    stat_type = 'Mean' if is_mean else 'Standard deviation'
    metric_type = f'{"prior" if "prior" in which_key else "posterior"} probability'
    
    this_score_df, this_score_arr = organize_scores(is_success = True, is_mean = is_mean, which_key = which_key)
    this_title = f'{stat_type} {metric_type} for cross-child analysis'
    
    this_figure = get_heatmap(this_title, this_score_arr)
    
    return this_figure
    
def get_heatmap(title, score_arr):

    figure = plt.figure()
    
    display_words = child_models.get_child_names()
    
    num_x_ticks = len(display_words)

    # For text annotations and color bar
    # 6/2 : https://www.pythonprogramming.in/heatmap-with-intermediate-color-text-annotations.html

    fig, ax = plt.subplots(figsize=(15, 15))

    plt.title(title)

    im = ax.imshow(score_arr, cmap = "YlGnBu")
    fig.colorbar(im)

    textcolors = ["k", "w"] 

    #6/2 hide ticks: https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/
    #6/2 rotation: https://www.delftstack.com/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/
    #6/2 xtick text: https://www.mathworks.com/help/matlab/creating_plots/change-tick-marks-and-tick-labels-of-graph-1.html

    plt.ylabel('Test items from child')
    plt.xlabel('Prior fine-tuned on child')
    
    plt.xticks(range(num_x_ticks), display_words, rotation = 45)
    plt.yticks(range(num_x_ticks), display_words)
    
    threshold = 6

    for i in range(len(display_words)):
        for j in range(len(display_words)):
            this_val = round(score_arr[i][j].item(), 3)
            ax.text(j, i, this_val, ha="center", va="center", color=textcolors[this_val > threshold])

    # End taken code

    return figure


def get_cross_type(data, prior):
    return f'data-{data}+prior_child-{prior}'


def get_cross_augmented_scores(data_child, prior_child):
    
    raw_scores = pd.read_pickle(utils_child.get_cross_path(data_child, prior_child))
    raw_scores['cross_type'] = get_cross_type(data_child, prior_child)
    raw_scores['data_child'] = data_child
    raw_scores['prior_child'] = prior_child
    
    return raw_scores
                   
                   
def load_all_scores():
    
    name_list = child_models.get_child_names()
    all_scores = pd.concat(
        [
            get_cross_augmented_scores(data_child, prior_child)
            for data_child in name_list
            for prior_child in name_list
        ]
    )
    
    return all_scores
    
    
def process_score_results(data_child, prior_child, which_key, is_mean = True):
    
    assert which_key in {'posterior_probability', 'prior_probability'}
    
    score_path = utils_child.get_cross_path(data_child, prior_child)
    scores = pd.read_pickle(score_path)
    
    ##################### TEMP CODE (only needed for my old copy of n = 2)
    
    if 'posterior_surprisal' in scores.columns:
        scores = scores.rename(columns={'posterior_surprisal' : 'posterior_probability',
                                          'prior_surprisal' : 'prior_probability'})
    
    ### ########## END TEMP CODE 
    
    scores = scores[scores.set == ('success')]
    
    # Match the R analyses "sem" function.
    stdev_match_r = lambda s : np.std(s, ddof = 1)
    
    agg_func = np.mean if is_mean else stdev_match_r
    
    stats = agg_func(-np.log2(scores[which_key]))
    
    return stats

    
def organize_scores(is_mean, is_success, which_key):
    
    results = defaultdict(list)

    name_list = child_models.get_child_names()
    for data_name in name_list:
        results[data_name] = [process_score_results(data_name, prior_name, is_mean = is_mean, which_key = which_key) for prior_name in name_list]
    
    results['Prior child name'] = name_list
    
    scores_arr = np.stack([np.array(results[name]) for name in name_list], axis = 0) 
    
    results_df = pd.DataFrame.from_records(results)
    
    return results_df, scores_arr