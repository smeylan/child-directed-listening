import os
from os.path import join, exists

import pandas as pd
import transformers 

# 6/21/21 From Dr. Meylan's yyy code
from pytorch_pretrained_bert import BertForMaskedLM
from transformers import BertTokenizer

from utils import transfomers_bert_completions

def get_model_dict():
    
    print('Note: This is all using old data from Dr. Meylan for now. Will need to update the model names, as well as the initialization data for the unigram models.') 
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    # Note all_old and meylan refer to the same split -- meylan means that Dr. Meylan trained the model and it's loaded from those weights.
    # all_old means that I trained it from Dr. Meylan's data
    
    all_model_dict = {
        'all_old/all_old/no_tags/0_context' : {
            'title': 'CHILDES BERT no speaker replication, same utt only', 
            'kwargs': load_models.get_all_data_models(with_tags = False).update({'context_width_in_utts' : 0}),
           'type' : 'BERT'
        },
        'all_old/all_old/with_tags/0_context' : {
           'title': 'CHILDES BERT, speaker tags, same utt only', 
           'kwargs': load_models.get_all_data_models(with_tags = True).update({'context_width_in_utts' : 0}),
           'type' : 'BERT',
        },
        'all_old/all_old/no_tags/20_context' : {
           'title': 'CHILDES BERT no speaker replication, +-20 utts context', 
           'kwargs': load_models.get_all_data_models(with_tags = True).update({'context_width_in_utts' : 20}),
           'type' : 'BERT',
        },
        'meylan/meylan/no_tags/20_context' : {'title': 'CHILDES BERT, +-20 utts context',
         'kwargs': {'modelLM': ft1_bertMaskedLM,
                    'tokenizer': ft1_tokenizer,
                    'softmax_mask': ft1_softmax_mask,
                    'context_width_in_utts': 20,
                    'use_speaker_labels':False
                   },
         'type': 'BERT'
        },
        'meylan/meylan/no_tags/0_context' : {'title': 'CHILDES BERT, same utt only',
         'kwargs': {'modelLM': ft1_bertMaskedLM,
                    'tokenizer': ft1_tokenizer,
                    'softmax_mask': ft1_softmax_mask,
                    'context_width_in_utts': 0,
                    'use_speaker_labels':False
                   },
         'type': 'BERT'
        },
        'meylan/meylan/no_tags/20_context' : {'title': 'Adult BERT, +-20 utts context',
        'kwargs': {'modelLM': adult_bertMaskedLM,
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'context_width_in_utts': 20,
                   'use_speaker_labels':False
                   },
         'type': 'BERT'
        },
        'meylan/meylan/no_tags/0_context' : {'title': 'Adult BERT, same utt only',
        'kwargs': {'modelLM': adult_bertMaskedLM,
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'context_width_in_utts': 0,
                   'use_speaker_labels':False
                   },
         'type': 'BERT'
        },        
        'meylan/meylan/unigram_childes' : {'title': 'CHILDES Unigram',
        'kwargs': {'child_counts_path': 'data/chi_vocab.csv',
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'vocab': initial_vocab
                   },
         'type': 'unigram'
        },
        'meylan/meylan/unigram_flat' : {'title': 'Flat Unigram',
        'kwargs': {'child_counts_path': None,
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'vocab': initial_vocab
                   },
         'type': 'unigram'
        }
    }
    
    return all_model_dict

def get_initial_vocab_info():
    
    # tokenize with the most extensive tokenizer, which is the one used for model #2
    initial_tokenizer = BertTokenizer.from_pretrained('model_output2')
    initial_tokenizer.add_tokens(['yyy','xxx']) #must maintain xxx and yyy for alignment,
    # otherwwise, BERT tokenizer will try to separate these into x #x and #x and y #y #y
    inital_vocab_mask, initial_vocab = transfomers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes.word)
    
    cmu_in_initial_vocab = cmu_2syl_inchildes.loc[cmu_2syl_inchildes.word.isin(initial_vocab)]

    return initial_vocab, cmu_in_initial_vocab

def get_model(model_path, with_tags, root_dir):
    
    # 6/21/21 Naming convention and general code from Dr. Meylan's original yyy code 
    
    word_info_all = get_cmu_dict_info(root_dir)
    word_info = word_info_all.word 
    model = BertForMaskedLM.from_pretrained(model_path)
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transfomers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'vocab' : vocab, 'use_speaker_labels' : with_tags }
    
    
def get_all_data_models(with_tags, root_dir):
    
    """
    Note: no_tags is the replication of Dr. Meylan's model_output 
    """
    
    # Load from local paths
    
    tag_type = 'no_tags' if not with_tags else 'with_tags'
    path = join(root_dir, join(join('models', 'meylan_model_output'), tag_type))
    
    # These are actually my replications of Dr. Meylan's original models
    # -- Dr. Meylan's models are located in "model_output" and "model_output2"
    
    return get_model(path, with_tags, root_dir)
    
    
def get_meylan_original_model(with_tags, root_dir):
    
    # Fine-tuned model 
    # Temporarily local for now
    
    model_name = '' if not with_tags else '2'
    model_path = join(root_dir, join('models', f'model_output{model_name}'))
    return get_model(model_path, with_tags, root_dir)


def get_cmu_dict_info(root_dir):
    
    # 6/21/21 Dr. Meylan's original yyy code
    cmu_in_childes = pd.read_csv(join(root_dir, 'phon/cmu_in_childes.csv'))
    cmu_2syl_inchildes = cmu_in_childes.loc[cmu_in_childes.num_vowels <=2]
    return cmu_2syl_inchildes


if __name__ == '__main__':
    
    get_all_data_models(False)
    get_all_data_models(True)