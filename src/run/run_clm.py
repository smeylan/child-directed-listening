import argparse
import transformers
import os
import time
import datetime
import pandas as pd
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


def sent_fixer(sent, use_tags):
    if not use_tags:
        sent = sent.replace('[CHI]','').replace('[CGV]', '').strip()                
        if len(sent) > 1:
            rsent = sent[0].title() + sent[1:]
        else:
            rsent = sent[0].title()
    
    else:
        split_sent = sent.split(' ')
        capitalized = split_sent[1][0].upper() + split_sent[1][1:]
        split_sent[1] = capitalized
        rsent = ' '.join(split_sent)
   
    return(rsent)

def sent_joiner(sents):
    return(' '.join(sents)) 

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|endoftext|>' + txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 
        

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type = str, help = 'Path to the text file to use for training')
    parser.add_argument('--validation_file', type = str, help = 'Path to the text file to use for validation')   
    parser.add_argument('--output_dir', type = str, help = 'Path to place the trained model')    
    parser.add_argument('--use_tags', type = int, help = 'Use tags in fine-tuning the model? 1=yes')    
    parser.add_argument('--batch_size', type = int, help = 'Number of batches to run in parallel on the GPU')    
        
    raw_args = parser.parse_known_args()[0]    
    this_model_args = vars(raw_args)


    this_model_args['task_phase'] = 'train'
    text_length = 100

    # initialize the tokenizer and add the special symbols
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|endoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    tokenizer.add_tokens("[CHI]")
    tokenizer.add_tokens("[CGV]")
    tokenizer.add_tokens("xxx") # can't be assigned phonetic transcript or orthographic transcript, e.g. microphone noise
    tokenizer.add_tokens("yyy") # can be assigned a phonetic transcript but not an orthographic transcript


    # add additional forms to the GPT-2 tokenizer that are in the BERT tokenizer
    print('Augmenting tokenizer with forms from BERT')

    # get the vocab
    import sys
    sys.path.append('.')
    sys.path.append('src/.')
    from src.utils import load_models
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab  = load_models.get_initial_vocab_info()
    # use initial_vocab

    tokens_to_add = [] 
    for bert_token in initial_vocab:
        if bert_token not in tokenizer.get_vocab().keys():
            if 'Ġ'+bert_token not in tokenizer.get_vocab().keys():
                tokens_to_add.append(bert_token)
                tokens_to_add.append(bert_token.title())

    tokenizer.add_tokens(tokens_to_add) 

    tokenizer.unique_no_split_tokens_dict = {word:1 for word in tokenizer.unique_no_split_tokens}


    if not os.path.exists(this_model_args['output_dir']):
        os.makedirs(this_model_args['output_dir'])
    
    tokenizer.save_pretrained(this_model_args['output_dir'])
    #Usine a dictionary for the lookup of tokenizer.unique_no_split_tokens allows for substantially faster tokenization

    #import pdb
    #pdb.set_trace() # make sure this captures "mommy", rooster, eetc.
    
    # test sentence = "mommy I want a rooster" -- 5 items, should have high indices for first and last

    # tokenizer.tokenize("[CHI] Mommy I want a rooster.")

    # WRONG: don't regenerate it from BERT, want the 7997 thing 
    # bert_tokenizer = GPT2Tokenizer.from_pretrained('bert-uncased')



    # # do the hack for speed:  add the contents of added_tokens.json to vocab.json, delete added_tokens.json, load using from_pretrained, and do a test tokenization

    # print('Move added tokens into the vocab...')

    # with open(os.path.join(this_model_args['output_dir'], 'added_tokens.json'), 'r') as added_tokens_file:
    #     gpt_added_tokens = json.load(added_tokens_file)
    
    # with open(os.path.join(this_model_args['output_dir'], 'vocab.json'), 'r') as vocab_file:
    #     gpt_vocab = json.load(vocab_file)

    # # write out the new vocab
    # for key, val in gpt_added_tokens.items():
    #     if key in gpt_vocab:
    #         raise ValueError(key + 'is already a key in the target tokenizer (GPT')
    #     gpt_vocab[key] = val

    # #remove the added_tokens again
    # with open(os.path.join(this_model_args['output_dir'], 'vocab.json'), 'w', encoding='utf-8') as vocab_file:
    #     json.dump(gpt_vocab, vocab_file, indent=4, ensure_ascii=False)
    # os.remove(os.path.join(this_model_args['output_dir'], 'added_tokens.json'))

    # # parallelize the tokenizer? 
    
    # tokenizer2 = GPT2Tokenizer.from_pretrained(os.path.join(this_model_args['output_dir']))
    

    # #'mommy' in tokenizer2.get_vocab().keys()
    # # what is the appropriate way to load this            

    #tokenizer.tokenize("[CHI] Mommy I want a rooster.")
    #tokenizer2.tokenize("[CHI] Mommy I want a rooster.")

    # Load the file to use for training
    training_file = this_model_args['train_file']
    train_df = pd.read_csv(training_file, header=None)  
    train_df.columns = ['sent']

    # Construct the dataset batches 
    train_num_inputs = int(np.ceil(train_df.shape[0] / text_length))
    group_idxs = np.repeat(range(0,train_num_inputs), repeats = text_length)

    # Fix the formatting of the sentences to look the most like the pretrained GPT model
    train_df['fixed'] = [sent_fixer(str(x), this_model_args['use_tags']) for x in train_df['sent']]
    train_df['group_idx'] = group_idxs[0:train_df.shape[0]]

    train_grouped_convos = train_df.groupby(['group_idx']).fixed.agg(sent_joiner).reset_index()

    # Bios: name for training data from the original approach
    training_sents = train_grouped_convos['fixed']
    print('Tokenizing....')
    training_dataset = GPT2Dataset(training_sents, tokenizer, max_length=768)
    print('Completed tokenizing the training sentences')

    # Load the file used for validation

    validation_file = this_model_args['validation_file']
    val_df = pd.read_csv(validation_file, header=None)  
    val_df.columns = ['sent']

    # Construct the dataset batches     
    val_num_inputs = int(np.ceil(val_df.shape[0] / text_length))
    val_group_idxs = np.repeat(range(0,val_num_inputs), repeats = text_length)

    # Fix the formatting of the sentences to look the most like the pretrained GPT model
    val_df['fixed'] = [sent_fixer(str(x), this_model_args['use_tags']) for x in val_df['sent']]
    val_df['group_idx'] = group_idxs[0:val_df.shape[0]]

    validation_grouped_convos = val_df.groupby(['group_idx']).fixed.agg(sent_joiner).reset_index()

    # Bios: name for training data from the original approach
    validation_sents = validation_grouped_convos['fixed']
    val_dataset = GPT2Dataset(validation_sents, tokenizer, max_length=768)

    # print('Check on the tags and punctuation')
    # import pdb
    # pdb.set_trace()

    # Split into training and validation sets
    train_size = len(training_dataset)
    val_size = len(val_dataset)    

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))


    # Initialize the dataloaders

    train_dataloader = DataLoader(
        training_dataset,  # The training samples.
        sampler = RandomSampler(training_dataset), # Select batches randomly
        batch_size = this_model_args['batch_size'] # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset, # The validation samples.
        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size = this_model_args['batch_size'] # Evaluate with this batch size.
    )

    # I'm not really doing anything with the config 
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    print('Loaded model....')

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    model.cuda()
    print('Moved model to the GPU....')

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    epochs = 5
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 100

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    optimizer = AdamW(model.parameters(),
      lr = learning_rate,
      eps = epsilon
    )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)

    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    total_t0 = time.time()

    training_stats = []

    model = model.to(device)
    

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        

            outputs = model(  b_input_ids,
                              labels=b_labels, 
                              attention_mask = b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 200,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):                    
                    print("{}: {}".format(i, tokenizer.decode(sample_output)))

                model.train()
                print('Saving the model') 
                model.save_pretrained(this_model_args['output_dir'])

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        
                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                 attention_mask = b_masks,
                                labels=b_labels)
              
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


