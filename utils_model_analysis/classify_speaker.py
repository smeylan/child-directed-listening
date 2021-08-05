
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from utils import load_models, transformers_bert_completions, split_gen
import numpy as np

import config

import os
from os.path import join, exists

import pandas as pd

import random

def analyze_completions(sentences, model, tokenizer, softmax_mask):
    
    sentence_results = {}
    
    for sentence in sentences:
        _, completions = transformers_bert_completions.bert_completions(sentence, model, tokenizer, softmax_mask)
        sentence_results[sentence] = completions
        
    return sentence_results

def analyze_model_tags(split = 'all', dataset = 'all'):
    
    tag_model = load_models.get_finetune_dict(split, dataset, True, 20)['kwargs']['modelLM']

    initial_tokenizer = load_models.get_primary_tokenizer()

    cmu_2syl_inchildes = load_models.get_cmu_dict_info()

    initial_vocab_mask, _ = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes.word)


    analysis_args = {
        'model' : tag_model,
        'tokenizer' : initial_tokenizer,
        'softmax_mask' : np.concatenate([initial_vocab_mask, np.array([30522, 30523])]),
    }
    
    val_file_path = join(config.finetune_dir, f'{split}/{dataset}/val.txt') # What to do here?
    
    mask_speaker = lambda s : ' '.join(['[MASK]'] + s.strip().split()[1:])
    
    with open(val_file_path, 'r') as f:
        sentences = f.readlines()
        
    random.shuffle(sentences)
    raw_sorted_sentences = sorted(sentences[:config.n_subsample])
    
    sorted_sentences = list(map(mask_speaker, raw_sorted_sentences))

    # Open the val file and get the appropriate sentences

    results = analyze_completions(sentences = sorted_sentences, **analysis_args)
    
    extract_chi_prob = lambda df: df[df.word == '[chi]']['prob'].item()
    extract_cgv_prob = lambda df: df[df.word == '[cgv]']['prob'].item()
    
    sorted_results = [ results[k] for k in sorted_sentences ]
    
    
    chi_probs = list(map(extract_chi_prob, sorted_results))
    cgv_probs = list(map(extract_cgv_prob, sorted_results))
    
    true_results = pd.DataFrame.from_records({
        'chi_prob' : chi_probs,
        'cgv_prob' : cgv_probs,
        'sentence' : raw_sorted_sentences,
    })
    
    this_split_loc = split_gen.get_split_folder(split, dataset, config.model_analyses_dir)
    
    results_path = join(this_split_loc, 'completions_classification.csv')
    true_results.to_csv(results_path)
    
    print(f"Saved results to: {results_path}")
     
    return true_results