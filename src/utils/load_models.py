import os
from os.path import join, exists
import pandas as pd
import numpy as np
import transformers 
import json
import copy
from transformers import BertTokenizer, BertForMaskedLM
from src.utils import configuration, transformers_bert_completions, split_gen, load_splits, child_models
config = configuration.Config()

def gen_finetune_model_args():

    finetune_model_args = []
    
    for model_arg_set in config.finetune_model_args:
                
        if model_arg_set['training_split'] in ('Providence','Providence-Age'):
            for use_tags in [True, False]:
                for context in config.context_list:
                    model_arg_set['use_tags'] = use_tags

                    model_arg_set['context_width'] = context
                    finetune_model_args.append(copy.copy(model_arg_set))

        else:
            for context in config.context_list:
                model_arg_set['use_tags'] = False
                model_arg_set['context_width'] = context
                finetune_model_args.append(copy.copy(model_arg_set))

    return finetune_model_args


def gen_adult_model_args():

    shelf_model_args = []
    
    for model_arg_set in config.shelf_model_args:        
        for context in config.context_list:
            model_arg_set['use_tags'] = False
            model_arg_set['context_width'] = context
            
            shelf_model_args.append(copy.copy(model_arg_set))        

    return shelf_model_args
    
def gen_unigram_args():
    
    load_args = []
    
    # Two unigram baselines
    for unigram_name in ['flat_unigram', 'data_unigram']:
        load_args.append({
                    'split_name': 'all', 
                    'dataset_name': 'all',  
                    'use_tags' : False,
                    'context_width' : 0,
                    'model_type': unigram_name
                })
    
    return load_args

def gen_shelf_model_args():
    
    return gen_adult_model_args() + gen_unigram_args()

def gen_all_model_args():
    
    """
    Generate all of the model arguments used in the analysis.
    Order: (split, dataset, tags, context, model_type)
    """
    
    return gen_adult_model_args() + gen_finetune_model_args() + gen_unigram_args()
     
    
def gen_model_title(split, dataset, is_tags, context_num, model_type, training_dataset=None):
    
    model_type_dict = {
        'childes' : 'CHILDES BERT',
        'adult' : 'Adult BERT',
        'flat_unigram' : 'Flat prior',
        'data_unigram' : 'CHILDES unigram',
        'switchboard': 'Switchboard BERT'
    }
    context_dict = {
        0 : 'same utt only',
        20 : '+-20 utts context',
    }

    dataset_dict = {
        'all' : '',
        'young' : 'younger children',
        'old' : 'older children'
    }
    dataset_dict.update({ k : k for k in child_models.get_child_names()})
    

    speaker_tags_dict = {
        True : 'with tags',
        False :  'without tags',
    }
    
    if training_dataset is None:
        model_title = f'{model_type_dict[model_type]} {speaker_tags_dict[is_tags]}, {dataset_dict[dataset]}, {context_dict[context_num]}'
    else:
        model_title = f'{model_type_dict[model_type]} {speaker_tags_dict[is_tags]}, {dataset_dict[dataset]}, {dataset_dict[training_dataset]}, {context_dict[context_num]}'
    
    return model_title
    
    

def get_tag_context_str(tags, context):
    
    assert not ((tags is None) ^ (context is None)), "Both with_tags and context_width should be none, or neither."
    
    tag_rep = 'na' if tags is None else ('with_tags' if tags else 'no_tags')
    context_piece = 'na' if context is None else context
    
    context_rep = f'{context_piece}_context'
    return tag_rep, context_rep
    

def get_model_id(split_name, dataset_name, use_tags, context_width, model_type):
    
    tag_str, context_str = get_tag_context_str(use_tags, context_width)
    model_id = '/'.join([split_name, dataset_name, tag_str, context_str, model_type])
    
    return model_id


def query_model_title(split, dataset, is_tags, context_num, model_type, training_dataset):
    return gen_model_title(split, dataset, is_tags, context_num, model_type, training_dataset)
    
    
def get_vocab_tok_modules():
    
    cmu_2syl_inchildes = get_cmu_dict_info()
    
    adult_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    adult_softmax_mask, adult_vocab = transformers_bert_completions.get_softmax_mask(adult_tokenizer, cmu_2syl_inchildes.word)
    
    initial_tokenizer = get_primary_tokenizer()
    
    _, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
    cmu_2syl_inchildes.word)  
    
    return adult_tokenizer, adult_softmax_mask, initial_tokenizer, initial_vocab
    
    
def get_shelf_dict(split, dataset, with_tags, context):
    """
    Adult BERT models, no finetuning
    """
    
    adult_bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
    adult_bertMaskedLM.eval()
    
    adult_tokenizer, adult_softmax_mask, _, _ = get_vocab_tok_modules()
    
    return {
            'title': gen_model_title('all', 'all', False, context, 'adult'), 
            'kwargs': {'modelLM': adult_bertMaskedLM,
                        'tokenizer': adult_tokenizer,
                        'softmax_mask': adult_softmax_mask,
                        'context_width_in_utts': context,
                       'use_speaker_labels':False
                       },
             'type': 'BERT'
         }



def get_data_unigram_dict(split, dataset, with_tags, context):
    
    adult_tokenizer, adult_softmax_mask, _, initial_vocab = get_vocab_tok_modules()
    
    return {
        'title': 'CHILDES Unigram',
        'kwargs': {'child_counts_path': f'{config.finetune_dir}/all/all/chi_vocab_train.csv',
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'vocab': initial_vocab,

                    # Added these default args 7/9/21 for compatibility with rest of the code
                    'context_width_in_utts': 0,
                    'use_speaker_labels': False,
                   },
         'type': 'unigram'
        }




def get_flat_unigram_dict(split, dataset, with_tags, context):
    
    adult_tokenizer, adult_softmax_mask, _, initial_vocab = get_vocab_tok_modules()
    
    return {
            'title': 'Flat Unigram',
            # Note that this assumes that flat prior = no information at all.
            # That means it doesn't observe any train/val split.
            # It just assigns uniform probability to every single word,
            # regardless of if that word appears in the train set or not. 
            'kwargs': {'child_counts_path': None,
                        'tokenizer': adult_tokenizer,
                        'softmax_mask': adult_softmax_mask,
                        'vocab': initial_vocab,
                       
                        # Added these default args 7/9/21 for compatibility with rest of the code
                        'context_width_in_utts': 0,
                        'use_speaker_labels': False,
                       },
             'type': 'unigram'
        } 
    

    
def get_finetune_dict(split, dataset, with_tags, context_width):
    
    this_dict = {
        'title' : gen_model_title(split, dataset, with_tags, context_width, 'childes'),
        'kwargs' : get_model_from_split(split, dataset,
                                        with_tags = with_tags),
        'type' : 'BERT',
    }
    this_dict['kwargs'].update({'context_width_in_utts' : context_width})
    
    return this_dict



def get_model_dict(split, dataset, with_tags, context, model_type):
    """
    Only for age/all splits. Child loading is in utils_child/child_models.py
    """
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    # for childes data.
    
    # If it's a pretrained BERT model with no finetuning, it has /shelf added to its model id
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    if model_type == 'childes': 
        model_dict = get_finetune_dict(split, dataset, with_tags, context)
    if model_type == 'switchboard': 
        model_dict = get_finetune_dict(split, dataset, with_tags, context)
    elif model_type == 'adult':
        model_dict = get_shelf_dict(split, dataset, with_tags, context)
    elif model_type == 'data_unigram': 
        model_dict = get_data_unigram_dict(split, dataset, with_tags, context)
    elif model_type == 'flat_unigram':
        model_dict = get_flat_unigram_dict(split, dataset, with_tags, context)
    
    # Update the tokenizers if needed.
    
    # 7/9/21: So childes doesn't need to re-add the tokens, and it works fine with the tokens, manually checked via prints    
    if model_type != 'childes':
        model_dict['kwargs']['tokenizer'].add_tokens(['[chi]','[cgv]'])

    # Always add tokens to the new models.
    model_dict['kwargs']['tokenizer'].add_tokens(['yyy','xxx']) #must maintain xxx and yyy for alignment,        
    
    return model_dict
    
    
    
def get_primary_tokenizer():

    shelf_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    shelf_tok.add_tokens(['[chi]', '[cgv]', 'yyy', 'xxx'])

    return shelf_tok
    
def get_initial_vocab_info(initial_tokenizer = None):
    
    """
    Satisfies all constraints of successes if syllables counted via CMU (not by actual_phonology or model_phonology) 
    """
    
    # tokenize with the most extensive tokenizer, which is the one used for model #2
    
    if initial_tokenizer is None:
        initial_tokenizer = get_primary_tokenizer() # default argument
        
    cmu_2syl_inchildes = get_cmu_dict_info()
    cmu_2syl_inchildes_reduced = np.unique(cmu_2syl_inchildes.word)
    
    inital_vocab_mask, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes_reduced)
        
    # get the indiices in cmu_2syl for each unique word type in initial_vocab
    cmu_indices_for_initial_vocab = [np.argwhere(cmu_2syl_inchildes.word.values == x).flatten() for x in initial_vocab]
    return initial_vocab, cmu_2syl_inchildes, cmu_indices_for_initial_vocab


def get_model_path(split, dataset, with_tags):
    
    """
    7/15/21: New function, just breaking up an old function.
    """
    
    tag_folder = 'with_tags' if with_tags else 'no_tags'
    this_path = join(split_gen.get_split_folder(split, dataset, config.model_dir), tag_folder)
    
    return this_path

def get_model_from_split(split, dataset, with_tags):
    
    """
    For getting models trained on OM2.
    7/15/21: Split out get model path orthogonally from this
    """
    
    this_path = get_model_path(split, dataset, with_tags)
    return get_model_from_path(this_path, with_tags)
    
    
def get_model_from_path(model_path, with_tags):
    
    word_info_all = get_cmu_dict_info()
    word_info = word_info_all.word 
    
    try:
        model = BertForMaskedLM.from_pretrained(model_path)
    except BaseException as e:
        print('Model loading failed. Does a model actually exist at '+model_path)
        print(e)
        raise ValueError('Terminating!')

    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transformers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'use_speaker_labels' : with_tags }
 

    

def get_cmu_dict_info():
    
    cmu_in_childes = pd.read_pickle(config.cmu_path)    
    return cmu_in_childes 

