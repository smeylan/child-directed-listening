import torch
import numpy as np
import pandas as pd
import scipy.stats
import copy
import time
from string import punctuation

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def bert_completions(text, model, tokenizer, softmax_mask):
  if type(text) is str:
    text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_index = tokenized_text.index('[MASK]')  
    num_masks = len(np.argwhere(np.array(tokenized_text) == '[MASK]'))    
  else: 
    indexed_tokens = text
    masked_index = indexed_tokens.index(tokenizer.convert_tokens_to_ids(['[MASK]'])[0])  
    num_masks = len(np.argwhere(np.array(indexed_tokens) == tokenizer.convert_tokens_to_ids(['[MASK]'])[0]))    

  if num_masks > 1:
      raise ValueError('Too many masks found, check data prepration')

  # Create the segments tensors.
  segments_ids = [0] * len(indexed_tokens)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)

  probs = softmax(predictions[0, masked_index].data.numpy()[softmax_mask])  
  words = np.array(tokenizer.convert_ids_to_tokens(range(predictions.size()[2])))[softmax_mask]
  
  word_predictions  = pd.DataFrame({'prob': probs[:,0], 'word':words[:,0]})
  word_predictions = word_predictions.sort_values(by='prob', ascending=False)    
  word_predictions['rank'] = range(word_predictions.shape[0])
  return(word_predictions)
  
def compare_completions(context, bertMaskedLM, tokenizer,
    candidates = None):
  continuations = bert_completions(context, bertMaskedLM, tokenizer)
  if candidates is not None:
    return(continuations.loc[continuations.word.isin(candidates)])
  else:
    return(continuations)

def get_completions_for_mask(utt_df, true_word, bertMaskedLM, tokenizer, softmax_mask, return_type='score') :
    gloss_with_mask =  tokenizer.convert_tokens_to_ids(['[CLS]']
        ) + utt_df.token_id.tolist() + tokenizer.convert_tokens_to_ids(['[SEP]'])
    completions = bert_completions(gloss_with_mask, bertMaskedLM, tokenizer, softmax_mask)
    if true_word in completions['word'].tolist():
        true_completion = completions.loc[completions['word'] == true_word].iloc[0]
        rank = true_completion['rank']
        prob =  true_completion['prob']
    else:
        rank = np.nan
        prob = np.nan
    entropy = scipy.stats.entropy(completions.prob, base=2)
    # [ ] here we can  a return type to get back the softmax
    if return_type == 'score':
        return(pd.DataFrame({'rank':[rank], 'prob': [prob], 'entropy':[entropy], 'num_tokens_in_context':[utt_df.shape[0]-1]}))
    elif return_type == 'completions':
        return(completions)
    else:
        raise ValueError('specify score or completions')

def get_stats_for_failure(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask,context_width_in_utts = None, use_speaker_labels = False, preserve_errors=False):
    '''dummy function because failures mostly just use get_completions_for_mask'''    
    t1 = time.time()
    utt_df = all_tokens.loc[all_tokens.id == selected_utt_id]
    
    if utt_df.shape[0] == 0:
        return None
    else:
        # convert the @ back to a mask for the target word
        utt_df.loc[utt_df.token == '@','token'] = '[MASK]'
        utt_df.loc[utt_df.token == '[MASK]','token_id'] = 103


    if context_width_in_utts is not None:        
        this_seq_utt_id = utt_df.iloc[0].seq_utt_id
        before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > this_seq_utt_id - (context_width_in_utts+1))]
        after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1))]
    
        # ready to use before_utt_df and after_utt_df    
        sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})
        before_by_sent = [x[1] for x in before_utt_df.groupby(['seq_utt_id'])]    
        i = n = 1
        while i < len(before_by_sent):
            before_by_sent.insert(i, sep_row)
            i += (n+1)
        before_by_sent_df = pd.concat(before_by_sent)

        after_by_sent = [x[1] for x in after_utt_df.groupby(['seq_utt_id'])]    
        i = n = 1
        while i < len(after_by_sent):
            after_by_sent.insert(i, sep_row)
            i += (n+1)
        after_by_sent_df = pd.concat(after_by_sent)

        utt_df = pd.concat([before_by_sent_df, sep_row, utt_df, sep_row, after_by_sent_df])

        if preserve_errors:
            #convert @ back to yyy for context items
            utt_df.loc[utt_df.token == '@','token'] = 'yyy'
            utt_df.loc[utt_df.token == 'yyy','token_id'] = tokenizer.convert_tokens_to_ids(['yyy'])[0]
            #convert @ back to xxx for context items
            utt_df.loc[utt_df.token == '$','token'] = 'xxx'
            utt_df.loc[utt_df.token == 'xxx','token_id'] = tokenizer.convert_tokens_to_ids(['xxx'])[0]

            if np.sum(utt_df.token == '[MASK]') > 1:
                print('Multiple masks in the surrounding context')
                import pdb
                pdb.set_trace()
        else: 
            utt_df.loc[utt_df.token == '@','token'] = '[MASK]'
            utt_df.loc[utt_df.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            #convert @ back to xxx for context items
            utt_df.loc[utt_df.token == '$','token'] = '[MASK]'
            utt_df.loc[utt_df.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    
    if not use_speaker_labels:
        # remove the speaker labels
        utt_df = utt_df.loc[~utt_df.token.isin(['[chi]','[cgv]'])]            
    
    if utt_df.shape[0] > 0:
        t2 = time.time()        
        #print('GPU retrieval time: '+str(time.time() - t2))
        #print('Total time: '+str(time.time() - t1))
        return(get_completions_for_mask(utt_df, None, bertMaskedLM, tokenizer, softmax_mask))
    else:
        print('Empty tokens for utterance '+str(selected_utt_id))
        return(None)


def get_stats_for_success(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask, return_type='score', context_width_in_utts=None, use_speaker_labels=False, preserve_errors=False):
    '''replace each token one at a time, sending it through get_completions_for_mask'''
    utt_df = all_tokens.loc[all_tokens.id ==     selected_utt_id]    
    
    if utt_df.shape[0] == 0:
        return(None)

    if not use_speaker_labels : 
        utt_df = utt_df.loc[~utt_df.token.isin(['[chi]','[cgv]'])]

    completions = [] 
    for mask_position in range(utt_df.shape[0] - 1): # skip the last entry --  don't need to score the punctuation
        utt_df_local = copy.deepcopy(utt_df)    
        utt_df_local.iloc[mask_position, np.argwhere(utt_df_local.columns == 'token')[0][0]] = ['[MASK]']
        utt_df_local.iloc[mask_position,np.argwhere(utt_df_local.columns == 'token_id')[0][0]] = tokenizer.convert_tokens_to_ids(['[MASK]'])

        #concatenate with preceding and following
        if context_width_in_utts is not None:
            this_seq_utt_id = utt_df_local.iloc[0].seq_utt_id
            before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > this_seq_utt_id - (context_width_in_utts+1))]
            after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1))]
        
            # ready to use before_utt_df and after_utt_df    
            sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})
            before_by_sent = [x[1] for x in before_utt_df.groupby(['seq_utt_id'])]    
            i = n = 1
            while i < len(before_by_sent):
                before_by_sent.insert(i, sep_row)
                i += (n+1)
            before_by_sent_df = pd.concat(before_by_sent)

            after_by_sent = [x[1] for x in after_utt_df.groupby(['seq_utt_id'])]    
            i = n = 1
            while i < len(after_by_sent):
                after_by_sent.insert(i, sep_row)
                i += (n+1)
            after_by_sent_df = pd.concat(after_by_sent)

            utt_df_local = pd.concat([before_by_sent_df, sep_row, utt_df_local, sep_row, after_by_sent_df])

            if preserve_errors:
                #convert @ back to yyy for context items
                utt_df_local.loc[utt_df_local.token == '@','token'] = 'yyy'
                utt_df_local.loc[utt_df_local.token == 'yyy','token_id'] = tokenizer.convert_tokens_to_ids(['yyy'])[0]
                #convert @ back to xxx for context items
                utt_df_local.loc[utt_df_local.token == '$','token'] = 'xxx'
                utt_df_local.loc[utt_df_local.token == 'xxx','token_id'] = tokenizer.convert_tokens_to_ids(['xxx'])[0]

                if np.sum(utt_df_local.token == '[MASK]') > 1:
                    print('Multiple masks in the surrounding context')
                    import pdb
                    pdb.set_trace()
            else: 
                #convert xxx and yyy in the context to masks
                utt_df_local.loc[utt_df_local.token == '@','token'] = '[MASK]'
                utt_df_local.loc[utt_df_local.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
                #convert @ back to xxx for context items
                utt_df_local.loc[utt_df_local.token == '$','token'] = '[MASK]'
                utt_df_local.loc[utt_df_local.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        if not use_speaker_labels:
            # remove the speaker labels
            utt_df_local = utt_df_local.loc[~utt_df_local.token.isin(['[chi]','[cgv]'])]

        completion = get_completions_for_mask(utt_df_local, 
            utt_df.iloc[mask_position].token, bertMaskedLM, tokenizer, softmax_mask,return_type) 
        if return_type == 'score':
            completion['mask_position'] = mask_position
            completion['token'] = utt_df.iloc[mask_position]['token']
            completion['utterance_id'] = selected_utt_id
        completions.append(completion)

    if return_type == 'score':
        scores = pd.concat(completions)            
        return(scores)
    
    elif return_type == 'completions':
        return(completions)
    else:
        raise ValueError('specify score or completions')
    
def get_softmax_mask(tokenizer):
    vocab_size = len(tokenizer.get_vocab())
    words = np.array(tokenizer.convert_ids_to_tokens(range(vocab_size)))
    mask = np.ones([vocab_size])
    mask[np.array(['##' in x for x in words])] = 0
    mask[np.array([any(p in x for p in punctuation) for x in words])] = 0
    return(np.argwhere(mask))

def compare_successes_failures(all_tokens, success_utts, yyy_utts, modelLM, tokenizer, softmax_mask, num_samples, num_context_utts, use_speaker_labels=True, preserve_errors=True):    
    print('Computing failure scores')
    import warnings
    warnings.filterwarnings('ignore')

    if num_samples is None:
        sample_yyy_utts = yyy_utts.utterance_id
    else:     
        sample_yyy_utts = np.random.choice(yyy_utts.utterance_id, num_samples)
    
    failure_scores = pd.concat([y for y in \
    [get_stats_for_failure(all_tokens, x,
    modelLM, tokenizer, softmax_mask, num_context_utts, use_speaker_labels, preserve_errors) for x in sample_yyy_utts] \
        if y is not None])
        
    print('Computing success scores')
    if num_samples is None:
        sample_success_utts = success_utts.utterance_id
    else:
        sample_success_utts = np.random.choice(success_utts.utterance_id, num_samples)   
    
    success_scores = pd.concat([y for y in  [ \
    get_stats_for_success(all_tokens, x,
        modelLM, tokenizer, softmax_mask, 'score', num_context_utts, use_speaker_labels, preserve_errors) for x in sample_success_utts] if y is not None])
    
    failure_scores['set'] = 'failure'
    success_scores['set'] = 'success'
    failure_and_success_scores = pd.concat([failure_scores, success_scores])

    warnings.filterwarnings('default')

    return(failure_and_success_scores)

def compare_successes_failures_unigram_model(all_tokens, success_utts, yyy_utts, tokenizer, softmax_mask, num_samples, child_counts_path):
    '''unigram model on CHILDES productions''' 

    # use tokenizer, softmax_mask to limit vocab 
    vocab = np.array(tokenizer.convert_ids_to_tokens(range(len(tokenizer.get_vocab()))))[softmax_mask][:,0]
    
    # child production stats only
    childes_counts = pd.read_csv(child_counts_path)
    childes_counts = childes_counts.loc[childes_counts.word.isin(vocab)]
    childes_counts['prob'] = childes_counts['count'] / np.sum(childes_counts['count'])
    
    # for successes, get the probability of all words
    sample_success_utts = np.random.choice(success_utts.utterance_id, num_samples)

    success_utt_contents = all_tokens.loc[all_tokens.id.isin(sample_success_utts)]
    success_utt_contents = success_utt_contents.loc[~success_utt_contents.token.isin(['[chi]'])]

    success_scores = success_utt_contents[['token']].merge(childes_counts, left_on='token', right_on='word')


    # entropy will be the same for all success and failure tokens
    constant_entropy = scipy.stats.entropy(childes_counts['prob'])
    success_scores['entropy'] = constant_entropy
    success_scores['set'] = 'success'
    
    failure_scores = pd.DataFrame({'entropy' : np.repeat(constant_entropy, num_samples)})
    failure_scores['set'] = 'failure'
    scores = pd.concat([success_scores, failure_scores])
    return(scores)




