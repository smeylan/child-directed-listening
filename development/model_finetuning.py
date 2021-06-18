import os
from os.path import join, exists

import transformers

from transformers import BertForMaskedLM, BertTokenizer

import datasets
from datasets import load_dataset

import importlib

# 6/3/21 Imports below from here:
# https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=zTgWPa9Dipk2
from transformers import Trainer, TrainingArguments


def get_datasets(tokenizer, with_tags):
    
    dataset_path = 'w-nicole/childes_data'
    
    # 6/9 : https://huggingface.co/docs/datasets/loading_datasets.html#cache-management-and-integrity-verifications
    # This can be used to check dataset after changes were made.
    
    #cache_path = os.path.join('/om2/user/wongn', 'temp_cache')
    #cache_path = os.path.join('./user/wongn', 'temp_cache3')
    #if not exists(cache_path): # This for verifications only.
    #    os.makedirs(cache_path)
        
    childes_data = load_dataset(dataset_path)#, cache_dir = cache_path)
    
    # 6/3: https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    
    clean_text = lambda text : text.strip('[CGV] ').strip('[CHI] ').strip('\n') if not with_tags else text.strip('\n')
    # If no omit tags just get rid of the trailing \n
    
    # Important note: For now, the text of the model does not receive the proper removal of speaker tags,
    # However I manually checked that the actual input_ids (the result of tokenization) does have the speaker tags removed. 
    
    # documentation on dataset attributes: https://huggingface.co/docs/datasets/exploring.html
     
    tokenize_function = lambda examples : tokenizer(list(map(clean_text, examples['text'])))
    
    tokenized_datasets = {}
    for phase in ['train', 'validation']:
        tokenized_datasets[phase] = childes_data[phase].map(tokenize_function, batched=True)
    
    # If you get NonMatchingSplitsSizesError
    # 6/3: https://github.com/huggingface/datasets/issues/215
    
    return tokenized_datasets

def get_bert_model():

    #2/20: https://huggingface.co/transformers/quickstart.html

    model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
    # 6/17 From Dr. Meylan
    tokenizer.add_tokens(['[chi]','[cgv]']) # Needed because of pre-tokenization of datasets.
    # Note that need lower case probably because bert is lowercase. Capitals won't work

    return model, tokenizer

def get_trainer(model, tokenizer, model_save_path, with_tags = True):
    
    
    this_datasets = get_datasets(tokenizer, with_tags)
    
    # Development of the training code 

    # Code is taken on 6/3/21 from here:
    # https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=zTgWPa9Dipk2

    # 6/17/21 : https://huggingface.co/transformers/main_classes/data_collator.html
    data_collator = transformers.data.data_collator.DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=model_save_path,
        overwrite_output_dir=True,
        num_train_epochs=100,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        tokenizer = tokenizer,
        train_dataset = this_datasets['train'],
        eval_dataset = this_datasets['validation'],
        
    )
    
    return trainer

if __name__ == '__main__':
    
    model_save_folder = '/om2/user/wongn/model_checkpoints/'
    #model_save_folder = './model_checkpoints/'
    
    model_save_path = join(model_save_folder, 'bert/remote_data/replication_no_tags')
    
    if not exists(model_save_folder):
        os.makedirs(model_save_folder)
            
    this_model, this_tokenizer = get_bert_model()
    
    trainer = get_trainer(this_model, this_tokenizer, model_save_path, with_tags = False)
    
    #Still from 6/3: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=VmaHZXzmkNtJ
    train_results = trainer.train()
    new_results = trainer.evaluate()
    
    
    
    
    
    
    



