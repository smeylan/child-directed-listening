import os
from os.path import join, exists

import pandas as pd
import transformers 

# 6/21/21 From Dr. Meylan's yyy code
from pytorch_pretrained_bert import BertForMaskedLM
from transformers import BertTokenizer

import transfomers_bert_completions


def get_model(model_path, with_tags, root_dir = '..'):
    
    # 6/21/21 Naming convention and general code from Dr. Meylan's original yyy code 
    
    word_info_all = get_cmu_dict_info(root_dir)
    word_info = word_info_all.word 
    model = BertForMaskedLM.from_pretrained(model_path)
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transfomers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'vocab' : vocab, 'use_speaker_labels' : with_tags }
    
    
def get_all_data_models(with_tags, root_dir = '..'):
    
    """
    Note: no_tags is the replication of Dr. Meylan's model_output 
    """
    
    # Load from local paths
    
    tag_type = 'no_tags' if not with_tags else 'with_tags'
    path = join(root_dir, join(join('models', 'meylan_model_output'), tag_type))
    
    # These are actually my replications of Dr. Meylan's original models
    # -- Dr. Meylan's models are located in "model_output" and "model_output2"
    
    return get_model(path, with_tags)
    
    
def get_meylan_original_model(with_tags, root_dir = '..'):
    
    # Fine-tuned model 
    # Temporarily local for now
    
    model_name = '' if not with_tags else '2'
    return get_model(join(root_dir, join('models', f'model_output{model_name}')), with_tags)


def get_cmu_dict_info(root_dir = '..'):
    
    # 6/21/21 Dr. Meylan's original yyy code
    cmu_in_childes = pd.read_csv(join(root_dir, 'phon/cmu_in_childes.csv'))
    cmu_2syl_inchildes = cmu_in_childes.loc[cmu_in_childes.num_vowels <=2]
    return cmu_2syl_inchildes


if __name__ == '__main__':
    
    get_all_data_models(False)
    get_all_data_models(True)