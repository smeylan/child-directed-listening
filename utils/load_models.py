import os
from os.path import join, exists

import pandas as pd
import transformers 

# 6/21/21 From Dr. Meylan's yyy code
from pytorch_pretrained_bert import BertForMaskedLM
from transformers import BertTokenizer

from utils import transfomers_bert_completions, split_gen

def get_model_dict(root_dir):
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    # Note all_old and meylan refer to the same split -- meylan means that Dr. Meylan trained the model and it's loaded from those weights.
    # all_old means that I trained it from Dr. Meylan's data
    
    cmu_2syl_inchildes = get_cmu_dict_info(root_dir)
    
    adult_bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
    adult_bertMaskedLM.eval()
    adult_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    adult_softmax_mask, adult_vocab = transfomers_bert_completions.get_softmax_mask(adult_tokenizer, cmu_2syl_inchildes.word)
    
    # From the original code, initial_vocab is declared with tokenizer from model 2
    # You should change this to be latest code eventually.
    print('Change the initial tokenizer to be based on latest trained models, eventually.')
    initial_tokenizer = get_meylan_original_model(with_tags = True,
                                                 root_dir = root_dir)['tokenizer']
    
    
    _, initial_vocab = transfomers_bert_completions.get_softmax_mask(initial_tokenizer,
    cmu_2syl_inchildes.word)  
    
    # Anything marked all_debug is for development purposes only -- it's me putting together
    # the right file types to develop the loading code and such.
    
    # Order: split name, dataset name, with tags
    args = [('all_debug', 'all_debug', True)] 
    
    titles = {
        'all_debug/all_debug/with_tags/0_context' : 'CHILDES BERT debug, same utt only -- debug',
        'all_debug/all_debug/with_tags/20_context' : 'CHILDES BERT debug, +-20 utts context -- debug'
    }
    
    all_model_dict = {}
    
    for arg_set in args:
        for context_width in [0, 20]:
            split, dataset, tags = arg_set
            tag_str = 'with_tags' if tags else 'no_tags'
            model_id = '/'.join([split, dataset, tag_str, f'{context_width}_context'])
            all_model_dict[model_id] = {
                'title' : titles[model_id],
                'kwargs' : get_model_from_split(split, dataset,
                                                with_tags = tags, 
                                                base_dir = root_dir).update({'context_width_in_utts' : context_width}),
                'type' : 'BERT',
            }
            
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
    initial_tokenizer = BertTokenizer.from_pretrained('model_output2')
    initial_tokenizer.add_tokens(['yyy','xxx']) #must maintain xxx and yyy for alignment,
    # otherwwise, BERT tokenizer will try to separate these into x #x and #x and y #y #y
    inital_vocab_mask, initial_vocab = transfomers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes.word)
    
    cmu_in_initial_vocab = cmu_2syl_inchildes.loc[cmu_2syl_inchildes.word.isin(initial_vocab)]

    return initial_vocab, cmu_in_initial_vocab


def get_model_from_split(split, dataset, with_tags, base_dir = '/home/nwong/chompsky/childes/child_listening_continuation/child-directed-listening'):
    """
    For getting models trained on OM2.
    """
    
    tag_folder = 'with_tags' if with_tags else 'no_tags'
    this_path = join(split_gen.get_split_folder('all_debug', 'all_debug', join(base_dir, 'models/new_splits')), tag_folder)
    
    return get_model_from_path(this_path, with_tags, base_dir)
    
    
def get_model_from_path(model_path, with_tags, root_dir):
    
    # 6/21/21 Naming convention and general code from Dr. Meylan's original yyy code 
    
    word_info_all = get_cmu_dict_info(root_dir)
    word_info = word_info_all.word 
    
    print(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transfomers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'vocab' : vocab, 'use_speaker_labels' : with_tags }
 
    
def get_meylan_original_model(with_tags, root_dir):
    
    # Fine-tuned model 
    # Temporarily local for now
    
    model_name = '' if not with_tags else '2'
    model_path = join(root_dir, join('models', f'model_output{model_name}'))
    return get_model_from_path(model_path, with_tags, root_dir)


def get_cmu_dict_info(root_dir):
    
    # 6/21/21 Dr. Meylan's original yyy code
    cmu_in_childes = pd.read_csv(join(root_dir, 'phon/cmu_in_childes.csv'))
    cmu_2syl_inchildes = cmu_in_childes.loc[cmu_in_childes.num_vowels <=2]
    return cmu_2syl_inchildes


if __name__ == '__main__':
    
    pass