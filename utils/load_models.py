import os
from os.path import join, exists

import pandas as pd
import transformers 

from transformers import BertTokenizer, BertForMaskedLM

from utils import transformers_bert_completions, split_gen, load_csvs
import config


def gen_finetune_model_args():

    load_bert_args = []
    
    for model_args in config.childes_model_args:
        
        this_split, this_dataset_name = model_args 
        
        for use_tags in [True, False]:
                
            for context in config.context_list:

                load_bert_args.append((this_split, this_dataset_name, use_tags, context, 'childes'))

    return load_bert_args 


def gen_adult_model_args():
    
    # Two adult baselines
    load_args = []
    baseline_args = ('all', 'all', False)
    for context in config.context_list:
        load_args.append(baseline_args + (context , 'adult'))
    
    return load_args
    
def gen_all_model_args():
    
    """
    Generate all of the model arguments used in the analysis.
    Order: (split, dataset, tags, context, model_type)
    """
    
    load_args = gen_adult_model_args() + gen_finetune_model_args()
        
    # Two unigram baselines
    for unigram_name in ['flat_unigram', 'data_unigram']:
        load_args.append(('all', 'all', False) + (0, unigram_name))
        
    return load_args
    
    
def gen_model_title(split, dataset, is_tags, context_num, model_type):
    
    model_type_dict = {
        'childes' : 'CHILDES BERT',
        'adult' : 'Adult BERT',
        'flat_unigram' : 'Flat prior',
        'data_unigram' : 'CHILDES unigram',
    }
    context_dict = {
        0 : 'same utt only',
        20 : '+-20 utts context',
    }

    dataset_dict = {
        'all' : '',
        'young' : 'younger children',
        'old' : 'older children', 
        'all_debug' : 'all_debug',
    }

    speaker_tags_dict = {
        True : 'with tags',
        False :  'without tags',
    }
    
    model_title = f'{model_type_dict[model_type]} {speaker_tags_dict[is_tags]}, {dataset_dict[dataset]}, {context_dict[context_num]}'
    
    return model_title
    
    

def get_tag_context_str(tags, context):
    
    assert not ((tags is None) ^ (context is None)), "Both with_tags and context_width should be none, or neither."
    
    tag_rep = 'na' if tags is None else ('with_tags' if tags else 'no_tags')
    context_piece = 'na' if context is None else context
    
    context_rep = f'{context_piece}_context'
    return tag_rep, context_rep
    

def get_model_id(split_name, dataset_name, with_tags, context_width, model_type):
    
    tag_str, context_str = get_tag_context_str(with_tags, context_width)
    model_id = '/'.join([split_name, dataset_name, tag_str, context_str, model_type])
    
    return model_id


def query_model_title(split, dataset, is_tags, context_num, model_type):
    return gen_model_title(split, dataset, is_tags, context_num, model_type)

    
def get_model_dict():
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    # for childes data.
    
    # If it's a pretrained BERT model with no finetuning, it has /shelf added to its model id
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    
    cmu_2syl_inchildes = get_cmu_dict_info()
    
    adult_bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
    adult_bertMaskedLM.eval()
    adult_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    adult_softmax_mask, adult_vocab = transformers_bert_completions.get_softmax_mask(adult_tokenizer, cmu_2syl_inchildes.word)
    
    # From the original code, initial_vocab is declared with tokenizer from model 2
    # You should change this to be latest code eventually.
    print('Change the initial tokenizer to be based on latest trained models, eventually.')
    initial_tokenizer = get_meylan_original_model(with_tags = True)['tokenizer']
    
    
    _, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
    cmu_2syl_inchildes.word)  
    
    # Anything marked all_debug is for development purposes only -- it's me putting together
    # the right file types to develop the loading code and such.
    
    # Order: split name, dataset name, with tags
    
    # Load the BERT-based models
    
    args = gen_finetune_model_args()
    
    # Development code only.
    # args = [('all_debug', 'all_debug', True, 0, 'childes'), ('age', 'old', True, 0, 'childes')]
    
    all_model_dict = {}
    
    for arg_set in args:
        split, dataset, tags, context_width, _ = arg_set
        model_id = get_model_id(split, dataset, tags, context_width, 'childes')
        all_model_dict[model_id] = {
            'title' : gen_model_title(split, dataset, tags, context_width, 'childes'),
            'kwargs' : get_model_from_split(split, dataset,
                                            with_tags = tags),
            'type' : 'BERT',
        }
        all_model_dict[model_id]['kwargs'].update({'context_width_in_utts' : context_width})

    # Load the normal BERT model
    
    prev_bert_dict = {}
    for context in config.context_list: # You should refactor this later to use your BERT args
        prev_bert_dict[f'all/all/no_tags/{context}_context/adult'] = {
            'title': gen_model_title('all', 'all', False, context, 'adult'), 
            'kwargs': {'modelLM': adult_bertMaskedLM,
                        'tokenizer': adult_tokenizer,
                        'softmax_mask': adult_softmax_mask,
                        'context_width_in_utts': context,
                       'use_speaker_labels':False
                       },
             'type': 'BERT'
         }
    
    
    # Load the unigram-based models
    
    unigram_dict = {
        'all/all/no_tags/0_context/data_unigram' : {
            'title': 'CHILDES Unigram',
            'kwargs': {'child_counts_path': f'{config.data_dir}/all/all/chi_vocab_train.csv',
                        'tokenizer': adult_tokenizer,
                        'softmax_mask': adult_softmax_mask,
                        'vocab': initial_vocab,
                       
                        # Added these default args 7/9/21 for compatibility with rest of the code
                        'context_width_in_utts': 0,
                        'use_speaker_labels': False,
                       },
             'type': 'unigram'
        },
        'all/all/no_tags/0_context/flat_unigram' : {
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
        },   
    }
    
    for model_id, model_dict in all_model_dict.items():
        
        # 7/9/21: So childes doesn't need to re-add the tokens, and it works fine with the tokens, manually checked via prints
        if 'childes' not in model_id:
            print('Adding the speaker tokens to this model dict')
            model_dict['kwargs']['tokenizer'].add_tokens(['[chi]','[cgv]'])
        
        print('********* CHECKING THE TOKENIZATION *****')
        this_tokenizer = model_dict['kwargs']['tokenizer']
        
        print(f'For model id {model_id}')
        print(this_tokenizer.convert_ids_to_tokens(this_tokenizer.encode("[CHI] i'm not going to do anything.")))
        print(this_tokenizer.convert_ids_to_tokens(this_tokenizer.encode('[CGV] back on the table if you wanna finish it.')))
    
    all_model_dict.update(unigram_dict)
    all_model_dict.update(prev_bert_dict)
        
    return all_model_dict


def get_initial_vocab_info():
    
    # tokenize with the most extensive tokenizer, which is the one used for model #2
    print("Change the tokenizer from model output2 to be dynamic if possible")
    
    cmu_2syl_inchildes = get_cmu_dict_info()
    initial_tokenizer = get_meylan_original_model(with_tags = True)['tokenizer']
    
    initial_tokenizer.add_tokens(['yyy','xxx']) #must maintain xxx and yyy for alignment,
    # otherwise, BERT tokenizer will try to separate these into x #x and #x and y #y #y
    inital_vocab_mask, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes.word)
    
    cmu_in_initial_vocab = cmu_2syl_inchildes.loc[cmu_2syl_inchildes.word.isin(initial_vocab)]

    return initial_vocab, cmu_in_initial_vocab


def get_model_from_split(split, dataset, with_tags):
    """
    For getting models trained on OM2.
    """
    
    tag_folder = 'with_tags' if with_tags else 'no_tags'
    this_path = join(split_gen.get_split_folder(split, dataset, config.model_dir), tag_folder)
    
    return get_model_from_path(this_path, with_tags)
    
    
def get_model_from_path(model_path, with_tags):
    
    word_info_all = get_cmu_dict_info()
    word_info = word_info_all.word 
    
    model = BertForMaskedLM.from_pretrained(model_path)
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transformers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'use_speaker_labels' : with_tags }
 
    
def get_meylan_original_model(with_tags):
    
    # Fine-tuned model 
    # Temporarily local for now
    
    model_name = '' if not with_tags else '2'
    model_path = join(config.meylan_model_dir, f'model_output{model_name}')
    return get_model_from_path(model_path, with_tags)


def get_cmu_dict_info():
    
    cmu_in_childes = load_csvs.load_csv_with_lists(config.cmu_path)
    
    # Added check for loading list objects as list objects, rather than str
    # Based on skimming the code this shouldn't affect the functionality
    # because the lists are not used in the code.
    
    cmu_2syl_inchildes = cmu_in_childes.loc[cmu_in_childes.num_vowels <=2]
    return cmu_2syl_inchildes 


