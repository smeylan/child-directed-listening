import os
import copy
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle5 as pickle

from src.utils import load_splits, configuration, paths
from mpl_toolkits.axes_grid1 import make_axes_locatable
config = configuration.Config()




def process_model_score(model_score, which_key, func):
    scores = copy.copy(model_score.loc[(model_score.set == 'success')])    

    if func == 'std':
        agg_func = lambda s : np.std(s, ddof = 1)        
    elif func == 'mean':
        agg_func = np.mean
    else:
        raise ValueError('func must be either `mean` or `std`')
    
    stat = agg_func(-np.log2(scores[which_key]))
    
    rdf = pd.DataFrame({
                which_key:stat, 
                'training_split': np.unique(model_score.training_split),        
                'training_dataset': np.unique(model_score.training_dataset),
                'test_split': np.unique(model_score.test_split),
                'test_dataset': np.unique(model_score.test_dataset),
                'n_samples': len(scores[which_key])
                        
    })
    return(rdf)

def assemble_child_scores_no_order(hyperparameter_set):
    
    """
    Load all of the non_child models for a given hyperparameter
    """

    task_name = 'analysis'
    task_phase = 'eval'         
    child_names = load_splits.get_child_names()

    # cross each child with the Providence testing data for each other child
    child_arg_list = []
    for training_child in child_names:      
        for test_child in child_names:
            child_arg_list.append(
                {'training_split': 'Providence-Child',
                'training_dataset': training_child,
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':True,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})

    
    # Pretends that Switchboard is a kid and cross with the Providence testing data for each other child
    for test_child in child_names:
        child_arg_list.append(
            {'training_split': 'Switchboard',
                'training_dataset': 'all',
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':False,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})


    # Pretends that Switchboard is a kid and cross with the Providence testing data for each other child
    for test_child in child_names:
        child_arg_list.append(
            {'training_split': 'Providence',
                'training_dataset': 'all',
                'test_split': 'Providence-Child',
                'test_dataset': test_child,
                'model_type':'BERT', 
                'use_tags':True,
                'context_width':20,
                'task_name': task_name,
                'n_samples' : config.n_across_time,
                'task_phase': task_phase})


    score_store = []
    
    for model_arg in child_arg_list:

                
        model_arg['n_samples'] = config.n_across_time
        
        # loading from 
        results_path = paths.get_directory(model_arg)    

        search_string = os.path.join(results_path, hyperparameter_set+'_run_models_across_time_*.pkl')
        print('Searching '+search_string)
        age_paths = glob.glob(search_string)
        
        single_model_store = []
        for this_data_path in age_paths:
            
            #data_df = pd.read_pickle(this_data_path)
            with open(this_data_path, "rb") as fh:
                data_df = pickle.load(fh)
                data_df['training_split'] = model_arg['training_split']
                data_df['training_dataset'] = model_arg['training_dataset']
                data_df['test_split'] = model_arg['test_split']
                data_df['test_dataset'] = model_arg['test_dataset']
                data_df['model_type'] = model_arg['model_type']
                data_df['model_type'] = model_arg['model_type']
            

                data_df['split'] = data_df.training_split + '_' + data_df.training_dataset
                data_df['model'] = paths.get_file_identifier(model_arg)

                single_model_store.append(copy.copy(data_df))

        if len(single_model_store) > 0:
            score_store.append(pd.concat(single_model_store))
                      
    return score_store


def get_score_heatmap(children_by_age, df, key, title, threshold, filename, cmap="YlGnBu", exclude_adult_datasets=False, _min=None, _max=None, secondary_df=None, normalize_by_child=False):

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    
    df['training_dataset_label'] = df.training_dataset
    df.loc[df.training_split == 'Providence', 'training_dataset_label'] = 'Providence'
    df.loc[df.training_split == 'Switchboard', 'training_dataset_label'] = 'Switchboard'
    df.sort_values(by=['test_dataset', 'training_dataset_label'])
    
    training_dataset_labels = copy.copy(children_by_age)
    test_datasets = copy.copy(children_by_age)
    
    if exclude_adult_datasets:
        pass
    else:
        training_dataset_labels += ['Providence']
    
    score_arr = np.empty([len(training_dataset_labels), len(test_datasets)])
    score_arr[:] = np.nan
    
    for i in range(len(training_dataset_labels)):
        for j in range(len(test_datasets)):
            query_df = df.loc[(df.training_dataset_label == training_dataset_labels[i]) & (df.test_dataset == test_datasets[j])]
            if query_df.shape[0] > 0:
                score_arr[i,j] = query_df[key]    
            else:
                pass    

    label_arr = copy.copy(score_arr)
    if normalize_by_child:
        # divide each row by the value for Providence
        #score_arr = (score_arr.T / score_arr[6,:][:,None]).T
        score_arr = (score_arr.T - score_arr[6,:][:,None]).T



    if secondary_df is not None:
        secondary_array = np.empty([len(training_dataset_labels), len(test_datasets)])
        secondary_array[:] = np.nan

        for i in range(len(training_dataset_labels)):
            for j in range(len(test_datasets)):
                query_df = secondary_df.loc[(secondary_df.training_dataset_label == training_dataset_labels[i]) & (secondary_df.test_dataset == test_datasets[j])]
                if query_df.shape[0] > 0:
                    secondary_array[i,j] = query_df[key]    
                else:
                    pass   
            
    figure = plt.figure()

    num_y_ticks = len(training_dataset_labels)
    num_x_ticks = len(test_datasets)
    
    fig, ax = plt.subplots(figsize=(15, 15))

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    im = ax.imshow(score_arr, cmap = cmap, vmin = _min, vmax = _max)
    
    if not exclude_adult_datasets:
        plt.axvline(x=5.5, color='black', linestyle='--')

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=2.5, pack_start = True)    
    plt.colorbar(im, fraction=0.01, pad=0, shrink=.75, cax=cax, orientation = 'horizontal')

    textcolors = ["k", "w"] 
    
    plt.xlabel('Test Data From Child')
    plt.ylabel('Fine-Tuning Data from Child/Dataset')

    plt.xticks(range(num_x_ticks), test_datasets, rotation = 45)
    plt.yticks(range(num_y_ticks), training_dataset_labels)
    plt.title(title, pad=20, fontsize=24)

    for i in range(len(training_dataset_labels)):
        for j in range(len(test_datasets)):                        
            if not np.isnan(score_arr[i][j]):                
                if secondary_df is None:
                    this_val = round(label_arr[i][j].item(), 3)
                elif np.isnan(secondary_array[i][j]):
                    this_val = round(label_arr[i][j].item(), 3)
                else:
                    this_val = str(round(label_arr[i][j].item(), 3))+'\n(' + str(round(secondary_array[i][j].item(), 3)) +')'
                ax.text(j, i, this_val, ha="center", va="center", color=textcolors[score_arr[i][j].item() > threshold])


    if not exclude_adult_datasets:
        fig.add_axes(cax)

    plt.rcParams.update({'font.size': 24})
    plt.savefig(filename)

    _min = np.amin(score_arr)
    _max = np.amax(score_arr)
    
    return figure, (_min, _max)


