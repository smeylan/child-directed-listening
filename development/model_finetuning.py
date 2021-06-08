import os
from os.path import join, exists

from transformers import BertForMaskedLM, BertTokenizer
import datasets
from datasets import load_dataset

import importlib

# Imports 6/3 from here:
# https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=zTgWPa9Dipk2

from transformers import DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments


def get_datasets():
    dataset_path = 'w-nicole/childes_data'
    childes_data = load_dataset(dataset_path)

def tokenize_datasets():
    
    # 6/3: https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb

    def tokenize_function(examples): 
        return tokenizer(examples["text"])

    tokenized_datasets = {}
    for phase in ['train', 'validation']:
        tokenized_datasets[phase] = childes_data[phase].map(tokenize_function, batched=True)
    

    
    # If you get NonMatchingSplitsSizesError
    # 6/3: https://github.com/huggingface/datasets/issues/215

def get_model():

    #2/20: https://huggingface.co/transformers/quickstart.html

    model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")

    return model, tokenizer

def train_model(model, tokenizer, model_save_path './models/bert_finetune'):
    
    
    # Development of the training code 

    # Code is taken on 6/3 from here:
    # https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=zTgWPa9Dipk2

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Try to memorize a single example as a sanity check

    training_args = TrainingArguments(
        output_dir=model_save_path,
        overwrite_output_dir=True,
        num_train_epochs=100,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer = tokenizer,
        train_dataset=tokenized_datasets['train'],
        eval_dataset = tokenized_datasets['validation'],
    )
    
    return trainer



