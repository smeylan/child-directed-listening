import os
from os.path import join, exists

import pandas as pd
import transformers 

from pytorch_pretrained_bert import BertForMaskedLM
from transformers import BertTokenizer

from utils import transformers_bert_completions, split_gen
import config


def get_model_id(split_name, dataset_name, with_tags, context_width):
    
    tag_str = 'with_tags' if with_tags else 'no_tags'
    model_id = '/'.join([split_name, dataset_name, tag_str, f'{context_width}_context'])
    return model_id

def query_model_title(split, dataset, is_tags, context_num):
    return config.model_titles[get_model_id(split, dataset, is_tags, context_num)]
    
    
def get_model_dict():
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    # Note all_old and meylan refer to the same split -- meylan means that Dr. Meylan trained the model and it's loaded from those weights.
    # all_old means that I trained it from Dr. Meylan's data
    
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
    args = [('all_debug', 'all_debug', True)]  # Need to change this to be a dynamic query later.
    
    all_model_dict = {}
    
    for arg_set in args:
        for context_width in config.context_list:
            split, dataset, tags = arg_set
            model_id = get_model_id(split, dataset, tags, context_width)
            all_model_dict[model_id] = {
                'title' : config.model_titles[model_id],
                'kwargs' : get_model_from_split(split, dataset,
                                                with_tags = tags),
                'type' : 'BERT',
            }
            all_model_dict[model_id]['kwargs'].update({'context_width_in_utts' : context_width})
            
    return all_model_dict

#         'meylan/meylan/no_tags/0_context' : {'title': 'CHILDES BERT, same utt only',
#          'kwargs': get_meylan_original_model(with_tags = False).update({'context_width_in_utts' : 0}),
#          'type': 'BERT'
#         },
#         'meylan/meylan/no_tags/20_context' : {'title': 'CHILDES BERT, same utt only',
#          'kwargs': get_meylan_original_model(with_tags = False).update({'context_width_in_utts' : 20}),
#          'type': 'BERT'
#         }, 
        # You will need to fix this to be on the train.py only? If vocab == the frequency of the vocab words.
#         'meylan/meylan/unigram_childes' : {'title': 'CHILDES Unigram',
#         'kwargs': {'child_counts_path': 'data/chi_vocab.csv',
#                     'tokenizer': adult_tokenizer,
#                     'softmax_mask': adult_softmax_mask,
#                     'vocab': initial_vocab
#                    },
#          'type': 'unigram'
#         },
#         'meylan/meylan/unigram_flat' : {'title': 'Flat Unigram',
#         'kwargs': {'child_counts_path': None,
#                     'tokenizer': adult_tokenizer,
#                     'softmax_mask': adult_softmax_mask,
#                     'vocab': initial_vocab
#                    },
#          'type': 'unigram'
#         }
#    }
    


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
    this_path = join(split_gen.get_split_folder('all_debug', 'all_debug', config.model_dir), tag_folder)
    
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
    
    cmu_in_childes = pd.read_csv(config.cmu_path)
    cmu_2syl_inchildes = cmu_in_childes.loc[cmu_in_childes.num_vowels <=2]
    return cmu_2syl_inchildes 


