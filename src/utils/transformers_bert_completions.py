import torch
import numpy as np
import pandas as pd
import scipy.stats
import gc
import copy
import time
from string import punctuation
import Levenshtein
from src.utils import data_cleaning, load_models
import srilm
import warnings

def softmax(x, axis=None):
    '''
        Compute softmax on a vector or matrix

        Args:
        x: vector or matrix
        axis: none if a vector, else dimension over which to compute the softmax
    '''
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return (y / y.sum(axis=axis, keepdims=True))


def bert_completions(text, model, tokenizer, softmax_mask):
  '''
        Retrieve completions for a single masked token from a huggingface-format BERT model

        Args:
        text: a string to tokenize with [MASK] inserted, but without [CLS] and [SEP] markers
        model: HuggingFace BERT model
        tokenizer: tokenizer to use the string
        softmax_mask: a vector of indices of vocabulary items over which to compute the softmax

        Returns:
        probs: a vector of prior probabilities for the completions, corresponding to the softmax_mask
        word_predictions: a dataframe with the highest to lowest ranked completions for the single masked tokens
  '''
  if type(text) is str:
    text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_index = tokenized_text.index('[MASK]')  
    num_masks = len(np.argwhere(np.array(tokenized_text) == '[MASK]'))    
  else: 
    indexed_tokens = text
    masked_index = indexed_tokens.index(tokenizer.convert_tokens_to_ids(['[MASK]'])[0])  
    num_masks = len(np.argwhere(np.array(indexed_tokens) == tokenizer.convert_tokens_to_ids(['[MASK]'])[0]))    

  if num_masks > 1:
      raise ValueError('Too many masks found, check data prepration')
  if num_masks == 0:
      raise ValueError('No mask found, check if truncation removed the target utterance token.')
    

  # Create the segments tensors.
  segments_ids = [0] * len(indexed_tokens)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # 7/1/21: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
  #torch.cuda.is_available = lambda : False
  
  if torch.cuda.is_available():
        tokens_tensor = tokens_tensor.cuda()
        segments_tensors = segments_tensors.cuda()
        model = model.cuda()
        #print('Using CUDA')
  else:
        pass
        #print('Not using CUDA')
    
  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)['logits']
   
  if torch.cuda.is_available():
    predictions = predictions.detach().cpu()
  
  # 7/9/21: I think tuple indexing is no longer supported.
  # From development session:
  # text of interest torch.Size([1, 7]) -> tokens_tensor.shape
  # logits shape torch.Size([1, 7, 30524]) -> predictions.shape

  # 7/9/21 assume predictions is the logit
   
  prior_probs = softmax(predictions[0][masked_index].data.numpy()[softmax_mask])  
   
  # the size is the size of the vocabulary -> the softmax array itself.
  words = np.array(tokenizer.convert_ids_to_tokens(range(predictions.size()[2])))[softmax_mask]
  
  word_predictions  = pd.DataFrame({'prior_prob': prior_probs, 'word':words})
    
  word_predictions = word_predictions.sort_values(by='prior_prob', ascending=False)    
  word_predictions['prior_rank'] = range(word_predictions.shape[0])
  return(prior_probs, word_predictions)
  
  
def compare_completions(context, bertMaskedLM, tokenizer, softmax_mask, candidates = None):
    '''
        Compare a set of possible completions for a context

        Args
        context: a string to tokenize with [MASK] inserted, but without [CLS] and [SEP] markers
        bertMaskedLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        candidates: return completions in a limited set of words  

        Returns:
        continuations: a dataframe with the highest- to lowest- ranked completions for the single masked tokens
    '''
    continuations = bert_completions(context, bertMaskedLM, tokenizer, softmax_mask)
    if candidates is not None:
        return(continuations.loc[continuations.word.isin(candidates)])
    else:
        return(continuations)

def get_completions_for_mask(utt_df, true_word, bertMaskedLM, tokenizer, softmax_mask) :    
    '''
        Get a completion for an utterance dataframe (for retrieveing completions from subsets of a data frame)

        Args:
        utt_df: data frame with `token` column, inculding a [MASK] token
        true_word: correct completion for the masked token (for computing surprisal) 
        bertMaskedLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        softmax map: a vector of indices of vocabulary items over which to compute the softmax

        Returns:
        priors: an n * m  matrix of prior probabilities (n tokens/predictions, m vocabulary, reflecting the softmax mask)
        completions: a list of dataframes with the rank and probability for each complation
        scores: a datagrame with n rows, containing surprisal (wrt true_word) and entropy 
    '''
    
    
    gloss_with_mask =  tokenizer.convert_tokens_to_ids(['[CLS]']
        ) + utt_df.token_id.tolist() + tokenizer.convert_tokens_to_ids(['[SEP]'])    
    
    priors, completions = bert_completions(gloss_with_mask, bertMaskedLM, tokenizer, softmax_mask)
        
    if true_word in completions['word'].tolist():
        true_completion = completions.loc[completions['word'] == true_word].iloc[0]
        prior_rank = true_completion['prior_rank']
        prior_prob =  true_completion['prior_prob']
    else:
        prior_rank = np.nan
        prior_prob = np.nan    
    
    entropy = scipy.stats.entropy(completions.prior_prob, base=2)
    
    
    # 7/29/21: https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas (casting)
    # 7/29/21: https://pbpython.com/pandas_dtypes.html (type names)
    # 7/29/21: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html (casting with dictionary)
    # For how to specify and call casts (.astype('float16'))
    
    # 7/29/21: https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html For information on sizes of integers etc.
    
    return_df = pd.DataFrame({'prior_rank':[prior_rank], 'prior_prob': [prior_prob], 'entropy':[entropy], 'num_tokens_in_context':[utt_df.shape[0]-1],
    'bert_token_id' : utt_df.loc[utt_df.token == '[MASK]'].bert_token_id}).astype({'num_tokens_in_context' : 'int32'})
    
    return(priors, completions, return_df)

    # end cites

def get_stats_for_failure(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask, context_width_in_utts, use_speaker_labels):
    
    '''
        Retrieve completions for a communicative failure (containing one mask corresponding to a yyy token)
        
        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_utt_id: utternce id to pull from all_tokens as the target utterance
        bertMaskedLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        softmax map: a vector of indices of vocabulary items over which to compute the softmax
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after [SEP]) a speaker identification token like [cgv] or [chi]        

        Returns: (returns directly from get_completions_for_mask)
        priors: an n * m  matrix of prior probabilities (n tokens/predictions, m vocabulary, reflecting the softmax mask)
        completions: a list of dataframes of length n with the rank and probability for each completion
        scores: a datagrame with n rows, containing surprisal (wrt true_word) and entropy 
    
    '''    
    t1 = time.time()    
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]
        
    if (utt_df.shape[0] == 0) or (utt_df.loc[utt_df.partition == 'yyy'].shape[0] == 0):
        return None
    else:
        
        # convert the @ back to a mask for the target word
        utt_df.loc[utt_df.partition == 'yyy','token'] = '[MASK]'
        utt_df.loc[utt_df.token == '[MASK]','token_id'] = 103

    if context_width_in_utts is not None:   

        # limit to the current transcript
        this_transcript_id = utt_df.iloc[0].transcript_id
        this_utt_order = utt_df.iloc[0].utterance_order

        if len(context_width_in_utts) == 1:
            # context width is symmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
        else:
            # context width is asymmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts[0]+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts[1] + 1)) & (all_tokens.transcript_id == this_transcript_id)]


    
        # ready to use before_utt_df and after_utt_df    
        sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})

        if before_utt_df.shape[0] > 0:
            before_by_sent = [x[1] for x in before_utt_df.groupby(['utterance_order'])]    
            i = n = 1
            while i < len(before_by_sent):
                before_by_sent.insert(i, sep_row)
                i += (n+1)
            before_by_sent_df = pd.concat(before_by_sent)
        else:
            before_by_sent_df = pd.DataFrame()

        if after_utt_df.shape[0] > 0:
            after_by_sent = [x[1] for x in after_utt_df.groupby(['utterance_order'])]    
            i = n = 1
            while i < len(after_by_sent):
                after_by_sent.insert(i, sep_row)
                i += (n+1)
            after_by_sent_df = pd.concat(after_by_sent)
        else:
            after_by_sent_df = pd.DataFrame()

        utt_df = pd.concat([before_by_sent_df, sep_row, utt_df, sep_row, after_by_sent_df])
     
        if np.sum(utt_df.token == '[MASK]') > 1:
            print('Multiple masks in the surrounding context')
            import pdb
            pdb.set_trace()        
    
    if not use_speaker_labels:
        # remove the speaker labels
        utt_df = utt_df.loc[~utt_df.token.isin(['[chi]','[cgv]'])]           
        
    # Prevent overly long sequences that will break BERT
    if utt_df.shape[0] > (512 - 2): # If there's no room to fit CLS and end token into BERT. 
        utt_df = data_cleaning.cut_context_df(utt_df)
           
    if utt_df.shape[0] > 0:
        t2 = time.time()        
        return(get_completions_for_mask(utt_df, None, bertMaskedLM, tokenizer, softmax_mask))
    else:
        print('Empty tokens for utterance '+str(selected_utt_id))
        return(None)

def get_stats_for_failure_gpt2(all_tokens, selected_utt_id, gptLM, tokenizer, vocab, contextualized, context_width_in_utts, use_speaker_labels):
    
    '''
        Retrieve completions for a communicative failure (containing one mask corresponding to a yyy token)
        
        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_utt_id: utternce id to pull from all_tokens as the target utterance
        gptLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        vocab: list of words to evaluate as completions, corresponding to softmax_mask
        contextualized: compute probabilities using a right context if 1, otherwise just left contexts
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after [SEP]) a speaker identification token like [cgv] or [chi]        

        Returns: (returns directly from get_completions_for_mask)
        priors: an n * m  matrix of prior probabilities (n tokens/predictions, m vocabulary, reflecting the softmax mask)
        completions: a list of dataframes of length n with the rank and probability for each completion
        scores: a datagrame with n rows, containing surprisal (wrt true_word) and entropy 
    
    '''    
    t1 = time.time()    
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]
        
    if (utt_df.shape[0] == 0) or (utt_df.loc[utt_df.partition == 'yyy'].shape[0] == 0):
        return None
    else:
        
        # convert the @ back to a mask for the target word
        utt_df.loc[utt_df.partition == 'yyy','token'] = '[MASK]'
        

    # build the following and preceding utts as context
    if context_width_in_utts is  None:   

        raise ValueError('context_width_in_utts must be passed')

    else:

        # limit to the current transcript
        this_transcript_id = utt_df.iloc[0].transcript_id
        this_utt_order = utt_df.iloc[0].utterance_order

        if len(context_width_in_utts) == 1:
            # context width is symmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
        else:
            # context width is asymmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts[0]+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts[1] + 1)) & (all_tokens.transcript_id == this_transcript_id)]


        if before_utt_df.shape[0] > 0:
            before_by_sent = [x[1] for x in before_utt_df.groupby(['utterance_order'])]    
            i = n = 1
            while i < len(before_by_sent):
                before_by_sent.insert(i, sep_row)
                i += (n+1)
            before_by_sent_df = pd.concat(before_by_sent)
        else:
            before_by_sent_df = pd.DataFrame()

        if after_utt_df.shape[0] > 0:
            after_by_sent = [x[1] for x in after_utt_df.groupby(['utterance_order'])]    
            i = n = 1
            while i < len(after_by_sent):
                after_by_sent.insert(i, sep_row)
                i += (n+1)
            after_by_sent_df = pd.concat(after_by_sent)
        else:
            after_by_sent_df = pd.DataFrame()

        utt_df = pd.concat([before_by_sent_df, sep_row, utt_df, sep_row, after_by_sent_df])


        # sanity check
        if np.sum(utt_df.token == '[MASK]') > 1:
            print('Multiple masks in the surrounding context')
            import pdb
            pdb.set_trace()        
    
        if not use_speaker_labels:
            # remove the speaker labels
            utt_df = utt_df.loc[~utt_df.token.isin(['[CHI]','[CGV]'])]           


        print('Utterances constructed!')
        import pdb
        pdb.set_trace()


        # handle the "contextualized" variable.

        if int(contextualized):
            # discard everything after the MASK in the 

            print('Reached data prep for contextualized  GPT-2')
            import pdb
            pdb.set_trace()

            utt_df['idx'] = np.range(0,utt_df.shape[0])  
            mask_idx = utt_df.loc[utt_df.token == '[MASK]'].idx
            utt_df =  utt_df.loc[ utt_df.idx <= mask_idx]
        else:
            print('Reached data prep for contextualized  GPT-2')
            import pdb
            pdb.set_trace()
            raise NotImplementedError
        
    # may need soemthing that handles long utterances
           
    if utt_df.shape[0] > 0:
        
        print('Reached probability computation for GPT-2')
        import pdb
        pdb.set_trace()
        

        test_phrase = ' '.join(utt_df.token)

        if contextualized:
            # this computes the completion probability for each item in vocab, considering the right context as continuations
            priors, completions = get_prob_dist_for_gpt_completion(test_phrase, vocab, gptLM)            
        
            print(completions.head(5).word.to_list())

        else: 
            # this computes the probability of the next item in the vocab, with no right continuations
            priors, completions = get_prob_dist_for_next_token(test_phrase, vocab, gptLM)
        # create a summary representation called "scores"
        entropy = scipy.stats.entropy(completions.prior_prob, base=2)        
        scores = pd.DataFrame({'prior_rank':[np.nan], 'prior_prob': [np.nan], 'entropy':[entropy], 'num_tokens_in_context':[utt_df.shape[0]-1],
        'bert_token_id' : utt_df.loc[utt_df.idx == mask_idx].bert_token_id})


        return(priors, completions, scores)

    else:
        print('Empty tokens for utterance '+str(selected_utt_id))
        return(None)


def get_stats_for_success_gpt2(all_tokens, selected_utt_id, gptLM, tokenizer, vocab, contextualized, context_width_in_utts, use_speaker_labels):
    
    '''
        Retrieve completions for all successes in a success_utt 
        
        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_utt_id: utternce id to pull from all_tokens as the target utterance
        gptLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        vocab: list of words to evaluate as completions, corresponding to softmax_mask
        contextualized: compute probabilities using a right context if 1, otherwise just left contexts
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after [SEP]) a speaker identification token like [cgv] or [chi]        

        Returns: (returns directly from get_completions_for_mask)
        priors: an n * m  matrix of prior probabilities (n tokens/predictions, m vocabulary, reflecting the softmax mask)
        completions: a list of dataframes of length n with the rank and probability for each completion
        scores: a datagrame with n rows, containing surprisal (wrt true_word) and entropy 
    
    '''    
    t1 = time.time()    
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]

    if not use_speaker_labels:
        # remove the speaker labels
        utt_df = utt_df.loc[~utt_df.token.isin(['[CHI]','[CGV]','[chi]','[cgv]'])]           
    

    # initialzie the vars that we will add to at each mask position
    priors = []
    completions = [] 
    stats = []

    if utt_df.loc[utt_df.partition == 'success'].shape[0] == 0:
        return(None)
    else:
        mask_positions = np.argwhere((utt_df['partition'] == 'success').to_numpy())   


    for mask_position in mask_positions.flatten().tolist(): 
        
        utt_df_local = copy.deepcopy(utt_df)    
        true_word  = utt_df_local.iloc[mask_position].token
        utt_df_local.iloc[mask_position, np.argwhere(utt_df_local.columns == 'token')[0][0]] = ['[MASK]']
        utt_df_local.iloc[mask_position,np.argwhere(utt_df_local.columns == 'token_id')[0][0]] = tokenizer.convert_tokens_to_ids(['[MASK]'])

        if context_width_in_utts is  None:   
            raise ValueError('context_width_in_utts must be passed')

        # limit to the current transcript
        this_transcript_id = utt_df.iloc[0].transcript_id
        this_utt_order = utt_df.iloc[0].utterance_order

        if type(context_width_in_utts) is int:
            # context width is symmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
        elif type(context_width_in_utts) is list:
            # context width is asymmetric
            before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > (this_utt_order - (context_width_in_utts[0]+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

            after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts[1] + 1)) & (all_tokens.transcript_id == this_transcript_id)]
        else:
            raise('Failed to parse context_width_in_utts')

        utt_df_local = pd.concat([before_utt_df, utt_df_local,  after_utt_df])
        if not use_speaker_labels:
            utt_df_local = utt_df_local.loc[~utt_df_local.token.isin(['[CHI]','[CGV]','[chi]','[cgv]'])]


        if not int(contextualized):            
            # of not contextualized, discard everything after the MASK in the data input
            utt_df_local['idx'] = range(0, utt_df_local.shape[0]) 
            mask_idx = utt_df_local.loc[utt_df_local.token == '[MASK]'].iloc[0].idx

            utt_df_local =  utt_df_local.loc[ utt_df_local.idx <= mask_idx]
            
        else:
            # if contextualized, keep the whole input
            pass
            
               
        if utt_df_local.shape[0] > 0:                        

            
            # these capitalization rules only work when the speaker labels are present
            if use_speaker_labels:
                by_word = utt_df_local.token.to_list()
                for i in range(1, len(by_word)):
                    if not by_word[i] ==  '[MASK]' and by_word[i-1] in ['[CGV]', '[CHI]', '[cgv]', '[chi]']:
                        by_word[i] = by_word[i].title()         
            else:                
                
                # check if the utterance id is different than the previous
                previous_utt = [0] +  utt_df_local.utterance_id.to_list()
                diff = np.array(utt_df_local.utterance_id.to_list() + [0]) !=  np.array(previous_utt)
                utt_df_local['new_utt'] = diff[0: utt_df_local.shape[0]] 
                utt_df_local['capitalized'] = utt_df_local.token                                                

                utt_df_local.loc[utt_df_local.new_utt, 'capitalized'] = [x.title()  for x in utt_df_local.loc[utt_df_local.new_utt, 'capitalized'] ]
                utt_df_local.loc[utt_df_local.capitalized == '[Mask]', 'capitalized'] = '[MASK]'  
                utt_df_local.loc[utt_df_local.capitalized == 'Xxx', 'capitalized'] = 'xxx' 
                utt_df_local.loc[utt_df_local.capitalized == 'Yyy', 'capitalized'] = 'yyy'  
                by_word = utt_df_local.capitalized.to_list() 

            test_phrase = ' '.join(by_word).replace(' ##','').replace(' ?','?').replace(' ...','...').replace(' .','.').replace(' !','!').replace('[cgv]','[CGV]').replace('[chi]','[CHI]').replace(" ' ","'").replace(' i ', ' I ').replace('Yyy','yyy').replace('Xxx','xxx')

            # also needs punctuation fixes, capturalization of first letter, capitalization of little  


            if int(contextualized):
                # this computes the completion probability for each item in vocab, considering the right context as continuations

                print('Reached probability computation for contextualized, success GPT-2')
                print(test_phrase)                
                this_priors, this_completions = get_prob_dist_for_gpt_completion(test_phrase, vocab, gptLM)            
                print(this_completions.head(5).word.to_list())
                import pdb
                pdb.set_trace()
            else: 

                print('Reached probability computation for non-contextualized, success GPT-2')
                # this computes the probability of the next item in the vocab, with no right continuations
                import pdb
                pdb.set_trace()
                this_priors, this_completions = get_prob_dist_for_next_token(test_phrase, vocab, gptLM)
                print(this_completions.head(5).word.to_list())


            # create a summary representation called "scores"            
            entropy = scipy.stats.entropy(this_completions.prior_prob, base=2)                
            
            prior_rank = this_completions.word.to_list().index(true_word)
            prior_prob = this_completions.loc[this_completions.word == true_word].prior_prob

            this_stats = pd.DataFrame({
            'prior_rank':prior_rank, 
            'prior_prob': prior_prob,
            'token': true_word,
            'entropy':entropy, 
            'num_tokens_in_context': utt_df_local.shape[0] -1,
            'bert_token_id': utt_df.iloc[mask_position]['bert_token_id']})
                

            priors.append(this_priors)
            completions.append(this_completions)
            stats.append(this_stats)

            
        else:
            print('Empty tokens for utterance '+str(stats))
            return(None)

    return(np.vstack(priors), completions, pd.concat(stats))


def flatten_list(nested_list):
    return([element for sublist in nested_list for element in sublist])


def try_gpt_batch_size(test_phrase, vocab, model, batch_size):
    try:        
        sentences = []    
        lowercase_surprisal_store = []

        print('Assessing lowercase completions...')
        i=0
        for word in vocab:
            i += 1
            test_phrase_temp = copy.copy(test_phrase)
            sent = test_phrase_temp.replace('[MASK]', word)         
            sentences.append(sent)

            # batch for faster inference, but too much will exceeed the memory on the GPU        
            if (i % batch_size == 0) or i == len(vocab):
                scores = model.surprise(sentences) 
                lowercase_surprisal_store.append([np.sum([y[1] for y in x if not np.isinf(y[1])]) for x in scores])    
                sentences = []            
                del scores
                gc.collect()
                torch.cuda.empty_cache()
                #print('Processed '+str(i)+ ' completions')
                
        lowercase_surprisals = flatten_list(lowercase_surprisal_store)        

        print('Assessing uppercase completions...')
        sentences = []    
        uppercase_surprisal_store = []
        i=0
        for word in vocab:
            i += 1
            test_phrase_temp = copy.copy(test_phrase)
            sent = test_phrase_temp.replace('[MASK]', word.title()) 
            sentences.append(sent)

            # batch for faster inference, but too much will exceeed the memory on the GPU
            if (i % batch_size == 0) or i == len(vocab):
                scores = model.surprise(sentences) 
                uppercase_surprisal_store.append([np.sum([y[1] for y in x if not np.isinf(y[1])]) for x in scores])    
                sentences = []            
                del scores
                gc.collect()
                torch.cuda.empty_cache()
                #print('Processed '+str(i)+ ' completions')
                
        surprisals = np.array(flatten_list(lowercase_surprisal_store) + flatten_list(uppercase_surprisal_store))
        
        log_likelihood = -1. * np.array(surprisals)
        exponentiated = np.exp(log_likelihood - np.max(log_likelihood))
        prob = exponentiated / np.sum(exponentiated)
        
        lowercase_probs = prob[0:len(vocab)]         
        uppercase_probs = prob[len(vocab):]
        assert(len(lowercase_probs) == len(uppercase_probs))

        normalized_prob =  lowercase_probs + uppercase_probs

        assert(len(normalized_prob) == len(vocab))

        return({"success": True, "normalized_prob":normalized_prob})
    except:
        print('FAILED (GPU OOM) with batch size of '+str(batch_size)+', backing off...')
        return({"success": False})


def get_prob_dist_for_gpt_completion(test_phrase, vocab, model):
    
    batch_size = 4000
    rv = {"success" :False}

    t1 = time.time()
    while not rv['success']: 
        print('Attempting batch size: '+str(batch_size))
        rv = try_gpt_batch_size(test_phrase, vocab, model, batch_size)
        batch_size = batch_size / 2

    normalized_prob = rv['normalized_prob']    
    t2 = time.time()
    print('Elapsed in model retrieval: '+str(t2 - t1))
    
    rdf = pd.DataFrame({
        'word': vocab,
        'score': -1. * np.log10(normalized_prob),
        'prior_prob': normalized_prob
    })  

    completions = rdf.sort_values(by = ['prior_prob'], ascending = False)
    completions.rank = range(completions.shape[0])

    priors = normalized_prob

    return(priors, completions)    


def get_stats_for_success(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask, context_width_in_utts=None, use_speaker_labels=False):
    '''
        Retrieve completions for ALL success tokens in a communicative success (converting each token to a mask in turn)        
        
        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_utt_id: utternce id to pull from all_tokens as the target utterance
        bertMaskedLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        softmax map: a vector of indices of vocabulary items over which to compute the softmax
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after [SEP]) a speaker identification token like [cgv] or [chi]        

        Returns: (returns directly from get_completions_for_mask)
        priors: an n * m  matrix of prior probabilities (n tokens/predictions, m vocabulary, reflecting the softmax mask)
        completions: a list of dataframes of length n with the rank and probability for each completion
        scores: a datagrame with n rows, containing surprisal (wrt true_word) and entropy 
    
    '''    
    
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]   
    
    if utt_df.shape[0] == 0:
        return(None)

    if not use_speaker_labels: 
        utt_df = utt_df.loc[~utt_df.token.isin(['[chi]','[cgv]'])]
        

    priors = []
    completions = [] 
    stats = []

    if utt_df.loc[utt_df.partition == 'success'].shape[0] == 0:
        return(None)
    else:
        mask_positions = np.argwhere((utt_df['partition'] == 'success').to_numpy())            
        #this had ben restricted to 0 in some cases

    for mask_position in mask_positions.flatten().tolist(): 
        utt_df_local = copy.deepcopy(utt_df)    
        utt_df_local.iloc[mask_position, np.argwhere(utt_df_local.columns == 'token')[0][0]] = ['[MASK]']
        utt_df_local.iloc[mask_position,np.argwhere(utt_df_local.columns == 'token_id')[0][0]] = tokenizer.convert_tokens_to_ids(['[MASK]'])

        #concatenate with preceding and following
        if context_width_in_utts is not None:

            this_transcript_id = utt_df.iloc[0].transcript_id
            this_utt_order = utt_df_local.iloc[0].utterance_order

            if len(context_width_in_utts) == 1:
                # context width is symmetrical

                before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > this_utt_order - (context_width_in_utts+1)) & (all_tokens.transcript_id == this_transcript_id)]
            
                after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
            else:
                # context width is asymmetrical

                # from eg 20 before
                before_utt_df = all_tokens.loc[(all_tokens.utterance_order < this_utt_order) & (all_tokens.utterance_order > this_utt_order - (context_width_in_utts[0]+1)) & (all_tokens.transcript_id == this_transcript_id)]
            
                after_utt_df = all_tokens.loc[(all_tokens.utterance_order > this_utt_order) & (all_tokens.utterance_order < this_utt_order + (context_width_in_utts[1] + 1)) & (all_tokens.transcript_id == this_transcript_id)]            

            # ready to use before_utt_df and after_utt_df    
            sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})
            if before_utt_df.shape[0] > 0:
                before_by_sent = [x[1] for x in before_utt_df.groupby(['utterance_order'])]    
                i = n = 1
                while i < len(before_by_sent):
                    before_by_sent.insert(i, sep_row)
                    i += (n+1)
                before_by_sent_df = pd.concat(before_by_sent)
            else:
                before_by_sent_df = pd.DataFrame()

            if after_utt_df.shape[0] > 0:
                after_by_sent = [x[1] for x in after_utt_df.groupby(['utterance_order'])]    
                i = n = 1
                while i < len(after_by_sent):
                    after_by_sent.insert(i, sep_row)
                    i += (n+1)
                after_by_sent_df = pd.concat(after_by_sent)
            else:
                after_by_sent_df = pd.DataFrame()

            utt_df_local = pd.concat([before_by_sent_df, sep_row, utt_df_local, sep_row, after_by_sent_df])
            
        if not use_speaker_labels:
            # remove the speaker labels
            utt_df_local = utt_df_local.loc[~utt_df_local.token.isin(['[chi]','[cgv]'])]
        
        # Prevent overly long sequences that will break BERT
        if utt_df_local.shape[0] > (512 - 2): # If there's no room to fit CLS and end token into BERT. 
            utt_df_local = data_cleaning.cut_context_df(utt_df_local)
        
        this_priors, this_completions, this_stats = get_completions_for_mask(utt_df_local, 
            utt_df.iloc[mask_position].token, bertMaskedLM, tokenizer, softmax_mask) 
        
        this_stats['mask_position'] = mask_position
        this_stats['token'] = utt_df.iloc[mask_position]['token']
        this_stats['utterance_id'] = selected_utt_id

        priors.append(this_priors)
        completions.append(this_completions)        
        stats.append(this_stats)    
    return(np.vstack(priors), completions, pd.concat(stats))

    
def get_softmax_mask(tokenizer, token_list):
    '''
        Generate a mask (numpy array of indices) of words in the tokenizer over which to compute the softmax

        Args:
        tokenizer: BERT tokenizer
        token_list: list of words to include in the softmax

        Returns: numpy array of indices in the vocabulary
    '''

    vocab_size = len(tokenizer.get_vocab())
    words = np.array(tokenizer.convert_ids_to_tokens(range(vocab_size)))
    mask = np.ones([vocab_size])
    # omit word parts
    mask[np.array(['##' in x for x in words])] = 0
    # omit punctuation
    mask[np.array([any(p in x for p in punctuation) for x in words])] = 0    
    token_list = set(token_list)
    # omit anything that isn't in the list of vocabulary items
    mask[~np.array([x in token_list for x in words])] = 0
    return(np.argwhere(mask)[:,0], words[np.argwhere(mask)][:,0])

def compare_successes_failures(all_tokens, selected_success_utts, selected_yyy_utts, modelLM, tokenizer, softmax_mask, context_width_in_utts, use_speaker_labels=False):    
    '''
        Get prior probabilities, completions, and scores from a BERT model for a list of utterance ids for communicative successes and a list of utterance ids for communicative failures 

        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_suuccess_utts: utterance ids known to be communicative successes
        selected_yyy_utts: utterance ids known to be communicative failures
        modelLM: masked language model in the transformers format
        tokenizer: BERT tokenizer
        softmax_mask: a vector of indices of vocabulary items over which to compute the softmax
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after [SEP]) a speaker identification token like [cgv] or [chi]        

        Returns: dictionary with two keys:            
            priors: n * m matrix of prior probabilities, where n is the number of communicative failures + communicative successes, and m is the size of the vocab identified by the softmax map. Failures are stacked on top of successes and are identified by a bert_token_id
            scores: a datframe of length n containing concatenated entropy scores, ranks, probabilities. Failures are stacked on top of successes and are identified by a bert_token_id
    '''
    
    print('Computing failure scores')
    
    warnings.filterwarnings('ignore')

    failure_priors_store = []
    failure_scores_store = []
       
    for sample_yyy_id in selected_yyy_utts:

        rv = get_stats_for_failure(all_tokens, sample_yyy_id, modelLM, tokenizer, softmax_mask, context_width_in_utts, use_speaker_labels) 

        if rv is None:
            pass
        else:
            prior, _, score = rv
            failure_scores_store.append(score)
            failure_priors_store.append(prior)
    
    rdict = {}    
    if len(failure_scores_store) > 0:
        failure_scores = pd.concat(failure_scores_store)
        failure_scores['set'] = 'failure'         
        failure_priors = np.vstack(failure_priors_store)
    else:        
        failure_scores = pd.DataFrame()
        failure_priors = None
    
    
    print('Computing success scores')
    success_priors_store = []
    success_scores_store = []

    
    for success_id in selected_success_utts:
    
        rv = get_stats_for_success(all_tokens, success_id, modelLM, tokenizer, softmax_mask, context_width_in_utts, use_speaker_labels) 
        if rv is None:
            pass
            #!!! may want to pass throug a placeholder here 
        else:
            prior, _, score = rv
            success_scores_store.append(score)
            success_priors_store.append(prior)
    
    if len(success_scores_store) > 0:
        success_scores = pd.concat(success_scores_store)    
        success_scores['set'] = 'success' 
        success_priors = np.vstack(success_priors_store)
    else:
        success_scores = pd.DataFrame()
        success_priors = None

    rdict['scores'] = pd.concat([failure_scores, success_scores])
    
    rdict['priors'] = np.vstack([x for x in [failure_priors, success_priors] if x is not None])

    warnings.filterwarnings('default')

    return(rdict)

def compare_successes_failures_gpt2(all_tokens, selected_success_utts, selected_yyy_utts, modelLM, tokenizer, vocab, contextualized, context_width_in_utts, use_speaker_labels=False):    
    '''
        Get prior probabilities, completions, and scores from a GPT-2 model for a list of utterance ids for communicative successes and a list of utterance ids for communicative failures 

        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_suuccess_utts: utterance ids known to be communicative successes
        selected_yyy_utts: utterance ids known to be communicative failures
        modelLM: masked language model in the transformers format
        tokenizer: GPT-2 tokenizer
        vocab: list of words to evaluate, corresponding to the vocabulary items in the softmax_mask variable given to the BERT models    
        contextualized: if 0, consider each word as a completion of the left context. If 1, consider vocab as completions for the current utterance
        context_width_in_utts: number of utterances on either side of the target utterance to feed as context to the model
        use_speaker_labels: is the first token of every utterance sequence (after punctuation, not including SEP) a speaker identification token like [CGV] or [CHI]        

        Returns: dictionary with two keys:            
            priors: n * m matrix of prior probabilities, where n is the number of communicative failures + communicative successes, and m is the size of the vocab identified by the softmax map. Failures are stacked on top of successes and are identified by a bert_token_id
            scores: a datframe of length n containing concatenated entropy scores, ranks, probabilities. Failures are stacked on top of successes and are identified by a bert_token_id
    '''
    
    print('Computing failure scores')
    
    warnings.filterwarnings('ignore')

    failure_priors_store = []
    failure_scores_store = []
       
    for sample_yyy_id in selected_yyy_utts:

        rv = get_stats_for_failure_gpt2(all_tokens, sample_yyy_id, modelLM, tokenizer, vocab, contextualized, context_width_in_utts, use_speaker_labels) 

        if rv is None:
            pass
        else:
            prior, _, score = rv
            failure_scores_store.append(score)
            failure_priors_store.append(prior)
    
    rdict = {}    
    if len(failure_scores_store) > 0:
        failure_scores = pd.concat(failure_scores_store)
        failure_scores['set'] = 'failure'         
        failure_priors = np.vstack(failure_priors_store)
    else:        
        failure_scores = pd.DataFrame()
        failure_priors = None
    
    
    print('Computing success scores')
    success_priors_store = []
    success_scores_store = []

    
    for success_id in selected_success_utts:
    
        rv = get_stats_for_success_gpt2(all_tokens, success_id, modelLM, tokenizer, vocab, contextualized, context_width_in_utts, use_speaker_labels) 
        if rv is None:
            pass
            #!!! may want to pass throug a placeholder here 
        else:
            prior, _, score = rv
            success_scores_store.append(score)
            success_priors_store.append(prior)
    
    if len(success_scores_store) > 0:
        success_scores = pd.concat(success_scores_store)    
        success_scores['set'] = 'success' 
        success_priors = np.vstack(success_priors_store)
    else:
        success_scores = pd.DataFrame()
        success_priors = None

    rdict['scores'] = pd.concat([failure_scores, success_scores])    
    rdict['priors'] = np.vstack([x for x in [failure_priors, success_priors] if x is not None])

    warnings.filterwarnings('default')

    return(rdict)


def compare_successes_failures_unigram_model(all_tokens, selected_success_utts, selected_yyy_utts, tokenizer, softmax_mask,  child_counts_path, vocab, context_width_in_utts, use_speaker_labels):
    '''
        Get prior probaiblities, completions, and scores from a unigram model (limiting to the vocab) or flat prior for a list of utterance ids for communicative successes and a list of utterance ids for communicative failures 

        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_suuccess_utts: utterance ids known to be communicative successes
        selected_yyy_utts: utterance ids known to be communicative failures        
        tokenizer: BERT tokenizer
        softmax_mask: a vector of indices of vocabulary items over which to compute the softmax
        child_counts_path: file to read with 'word' and 'count' in the column titles to use for unigram counts. If set to None, return a uniform prior
        raw_vocab: a string representation of the vocab without CMU/BERT limitations
        
        context_width_in_utts, use_speaker_labels : Ignored, used for compatibility with expected model dictionary elements in BERT models.

        Returns: dictionary with two keys:            
            priors: n * m matrix of prior probabilities, where n is the number of communicative failures + communicative successes, and m is the size of the vocab identified by the softmax map. Failures are stacked on top of successes and are identified by a bert_token_id
            scores: a datframe of length n containing concatenated entropy scores, ranks, probabilities. Failures are stacked on top of successes and are identified by a bert_token_id
    '''    
    
    unigram_model = pd.DataFrame({'word':vocab}) # because we start with the vocab, probability mass is just over the vocab
    
    if child_counts_path is not None: 
        childes_counts = pd.read_csv(child_counts_path)    
        unigram_model = unigram_model.merge(childes_counts, how='left')
        unigram_model = unigram_model.fillna(0) 
        unigram_model['count'] = unigram_model['count'] + .01 #additive smoothing
        
        unigram_model['prior_prob'] = unigram_model['count'] / np.sum(unigram_model['count'])
        unigram_model_ordered_by_initial_vocab = copy.copy(unigram_model)
        unigram_model = unigram_model.sort_values(by='prior_prob', ascending=False)
    
    else:
        # build a flat prior: assign all words equal probability
        unigram_model['prior_prob'] = 1/unigram_model.shape[0]        
        unigram_model_ordered_by_initial_vocab = copy.copy(unigram_model)

    # for successes, get the probability of all words        
    # Only score the success tokens
    success_utt_contents = all_tokens.loc[(all_tokens.utterance_id.isin(selected_success_utts))
                                         & (all_tokens.partition == 'success')]
    
    # Always override tags specification to False
    success_utt_contents = success_utt_contents.loc[~success_utt_contents.token.isin(['[chi]', '[cgv]'])]
    
    success_scores = success_utt_contents[['token','bert_token_id']].merge(unigram_model, left_on='token', right_on='word', how='left')
    # this gets the probability of the true word, but not the rank -- that would need to be calculated separately

    # test this stuff below
    if child_counts_path is not None:
        # but the unigram model != the initial_vocab list, so we can't take the rank directly
        success_scores['prior_rank'] = [np.where(unigram_model.word == x)[0][0] for x in success_scores.word]
    else:
        success_scores['prior_rank'] = int(len(vocab) / 2)
    # need to get the rank of the token in the unigram_model table
    # if childes_counts_path is null (ie unigram), then get take 1/2 of the value
    

    # entropy will be the same for all success and failure tokens under the unigram and flat models
    constant_entropy = scipy.stats.entropy(unigram_model['prior_prob'])
    success_scores['prior_entropy'] = constant_entropy
    success_scores['set'] = 'success'
    
    # need to retrieve the failures in the same way so I can limit by bert_token_id
    failure_scores = all_tokens.loc[(all_tokens.utterance_id.isin(selected_yyy_utts)) &
      (all_tokens.partition == 'yyy') ][['token','bert_token_id']]

    failure_scores['prior_entropy'] = constant_entropy
    failure_scores['set'] = 'failure'

    prior_vec = unigram_model_ordered_by_initial_vocab['prior_prob'].to_numpy()
    
    rdict = {}
    prior_list =[]
    scores_list = []
    
    if failure_scores.shape[0] > 0:
        prior_list.append(np.vstack([prior_vec for x in range(failure_scores.shape[0])]))
        scores_list.append(failure_scores )

    if success_scores.shape[0] > 0:
        prior_list.append(np.vstack([prior_vec for x in range(success_scores.shape[0])]))
        scores_list.append(success_scores)
   
    if len(scores_list) == 0:
        raise ValueError('Neither successes nor failures in given ids')

    rdict['priors'] = np.vstack(prior_list)
    rdict['scores'] = pd.concat(scores_list) 
    
    return(rdict)


def compare_successes_failures_ngram_model(all_tokens, selected_success_utts, selected_yyy_utts, vocab, ngram_path, order, contextualized):
    '''
        Get prior probaiblities, completions, and scores from a SRILM-style ngram model for a list of utterance ids for communicative successes and a list of utterance ids for communicative failures 

        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_suuccess_utts: utterance ids known to be communicative successes
        selected_yyy_utts: utterance ids known to be communicative failures                
        vocab:  words to test as priors under the model -- corresponding to positions in the softmax mask vector        
        ngram_path: path to a SRILM model
        order: highest order to use for retrieval with the ngram model        

        Returns: dictionary with two keys:            
            priors: n * m matrix of prior probabilities, where n is the number of communicative failures + communicative successes, and m is the size of the vocab identified by the softmax map. Failures are stacked on top of successes and are identified by a bert_token_id
            scores: a datframe of length n containing concatenated entropy scores, ranks, probabilities. Failures are stacked on top of successes and are identified by a bert_token_id
    '''    
    
    #!!! like the BERT verwsion of this, and unlike the unigram / uniform versions, need to grab the tokens in the context    

    ngram_lm = srilm.LM(ngram_path, lower=True)
    
    print('Computing failure scores...')
    failure_priors_store = []
    failure_scores_store = []
  

    for sample_yyy_id in selected_yyy_utts: 
        
        rv = get_ngram_failure_stats(all_tokens, sample_yyy_id, vocab, ngram_lm,  order, contextualized)

        prior, score = rv        
        failure_scores_store.append(score)
        failure_priors_store.append(prior)

    rdict = {}    
    if len(failure_scores_store) > 0:
        failure_scores = pd.concat(failure_scores_store)
        failure_scores['set'] = 'failure'         
        failure_priors = np.vstack(failure_priors_store)
    else:        
        failure_scores = pd.DataFrame()
        failure_priors = None
    
    print('Computing success scores...')
    
    success_priors_store = []
    success_scores_store = []
    continuations_store = []
    
    #for success_id in [16989621]:
    for success_id in selected_success_utts:
        print(success_id)        
        rv = get_ngram_success_stats(all_tokens, success_id, vocab, ngram_lm,  order, contextualized) 
        if rv is None:
            pass
            #!!! may want to pass through a placeholder here 
        else:
            prior, continuations, score = rv
            success_priors_store.append(prior)
            continuations_store.append(continuations)
            success_scores_store.append(score)
            
            


    if len(success_scores_store) > 0:
        success_scores = pd.concat(success_scores_store)    
        success_scores['set'] = 'success' 
        success_priors = np.vstack(success_priors_store)
    else:        
        success_scores = pd.DataFrame()
        success_priors = None

    rdict['continuations'] = continuations_store    
    rdict['scores'] = pd.concat([failure_scores, success_scores])        
    rdict['priors'] = np.vstack([x for x in [failure_priors, success_priors] if x is not None])    

    warnings.filterwarnings('default')

    return(rdict)

def srilm_word_by_word_prob(utt_df, n, lm, append_sos_bos, exclude_special, vocab, partition, print_debug=False):

    # sentence: sentence as a space-separated string
    # n: order of the n-gram
    # lm: SRILM model to use
    # append_sos_bos: add <s> and </s>? Usually yes, but no for continuous models
    # exclude_special: drop sos and transition tokens when reporting probabilities
    # print_debug: show current and preceding
    
    # same output as 
    # `ngram -lm /shared_hd0/corpora/BNC/SRILM/BNC_merged.LM -tolower -ppl  fox.txt -debug 2`    

    if not partition in ['success','yyy']:
        raise ValueError('Partition must be success or failure')

    # make a whitespace-separated
    whitespace_tokenized = ['<s>']+' '.join(utt_df.token).replace(' ##','').split(' ')+ ['</s>']
    whitespace_tokenized_for_search = copy.copy(whitespace_tokenized)
    
    # need to align each success in the utt_df with the appropriate preceding words in a space-separated version of the sentence

    prior_vecs = []
    continuations = []
    sentence_token_list = [] 
    bert_token_ids = []

    for i in np.argwhere((utt_df['partition'] == partition).to_numpy()).flatten().tolist():
        current = utt_df.iloc[i].token
        sentence_token_list.append(current)     
        bert_token_ids.append(utt_df.iloc[i].bert_token_id)
        
        #if there are a multiple occurences of a word, replaces them with MATCHED so that they can't be matched anymore
        loc_in_whitespace = whitespace_tokenized_for_search.index(current)
        whitespace_tokenized_for_search[loc_in_whitespace] = 'MATCHED'

        preceding = whitespace_tokenized[max((i+1)-(n-1),0):(i+1)] # adjust +1 for sos
        preceding_reversed = preceding[::-1]

        prior_prob_per_candidate_in_vocab = np.zeros(len(vocab))
        j = -1
        for candidate_word in vocab:
            j += 1
            prior_prob_per_candidate_in_vocab[j] = 10. ** lm.logprob_strings(candidate_word, preceding_reversed)

        continuations_for_word = pd.DataFrame({
            'word': vocab, 
            'probability':  prior_prob_per_candidate_in_vocab
        }).sort_values(by='probability', ascending=False)


        continuations.append(continuations_for_word.copy())
        prior_vecs.append(prior_prob_per_candidate_in_vocab)            

    rdf = pd.DataFrame({'token':sentence_token_list, 'prior_vec': prior_vecs,
        'continuations': continuations, 'bert_token_id':bert_token_ids})

    rdf['index'] = range(rdf.shape[0])

    excludes = ['</s>','<s>','[cgv]', '[chi]', '[cgv]2[cgv]', 
               '[chi]2[chi]', '[chi]2[cgv]','[cgv]2[chi]']
    
    if exclude_special:
        rdf = rdf.loc[~rdf.word.isin(excludes)]
    
    return(rdf)


def get_prob_dist_for_masked_token_fast(tokens, lm, vocab):
    
    # re-tokenize on whitespace and get rid of the word boundaries
    tokens = (' '.join(tokens).replace(' ##','')).split(' ')
    mask_idx = tokens.index("[mask]")
    
    # prior doesn't care if this is a success -- that is only in the score
    prior_vec = np.zeros(len(vocab))
    for i in range(len(vocab)):    
        temp_tokens = tokens.copy()
        temp_tokens[mask_idx] = vocab[i]               
        prior_vec[i] = 10. ** lm.total_logprob_strings(temp_tokens)
    
    prior_vec[np.isinf(prior_vec)] = 0
    # normalize it
    prior_vec = prior_vec / np.sum(prior_vec)


    continuations_table = pd.DataFrame({
        'word': vocab,
        'probability': prior_vec
    })
    continuations_table = continuations_table.sort_values(by=['probability'], ascending=False)

    return(prior_vec, continuations_table)



def get_ngram_failure_stats(all_tokens, selected_utt_id, vocab, ngram_lm,  order, contextualized):

    # get_stats_for_failure looks for the one yyy item in the selected_utt_id
    print(selected_utt_id)
    
    # replace the yyy with mask
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]
    if (utt_df.shape[0] == 0) or (utt_df.loc[utt_df.partition == 'yyy'].shape[0] == 0):
        return None
    else:
        utt_df.loc[utt_df.partition == 'yyy','token'] = '[mask]'

    if int(contextualized):
        # marginalize across all of the possible completions 
        tokens  = utt_df.token.to_list()        
        prior_vec_for_failure, continuations_table = get_prob_dist_for_masked_token_fast(tokens, ngram_lm, vocab)

    else: 

        # just get the probability distribution for the next word (no marginalization)
        word_by_word = srilm_word_by_word_prob(utt_df, order, ngram_lm, True, False, vocab, 
            'yyy', print_debug=False)
        prior_vec_for_failure =  word_by_word.loc[word_by_word.token == '[mask]']['prior_vec'][0]

    prior_rank = np.nan
    prior_prob = np.nan  
    entropy = scipy.stats.entropy(prior_vec_for_failure, base=2)  

    score_df = pd.DataFrame({
        'prior_rank':[prior_rank], 
        'prior_prob': [prior_prob],
        'entropy':[entropy], 
        'num_tokens_in_context':[utt_df.shape[0]-1],
        'bert_token_id' : utt_df.loc[utt_df.token == '[mask]'].bert_token_id,
        'num_tokens_in_context' : np.nan })

    return(prior_vec_for_failure, score_df)


def get_ngram_success_stats(all_tokens, selected_utt_id, vocab, ngram_lm,  order, contextualized):

    excludes = ['[chi]','[cgv]']

    # replace the yyy with mask
    utt_df = all_tokens.loc[all_tokens.utterance_id == selected_utt_id]
    if (utt_df.shape[0] == 0):
        return None    

    if int(contextualized):

        # iterating through the words in the utt_df with a mask
        prior_vecs = []
        continuations = [] 

        for i in np.argwhere((utt_df['partition'] == 'success').to_numpy()).flatten().tolist():
            
            tokens = utt_df.token.tolist()
            
            if tokens[i] in excludes:
                continue
            else:                
                tokens[i] = '[mask]'
                prior_vec, continuations_table = get_prob_dist_for_masked_token_fast(tokens, ngram_lm, vocab)
                prior_vecs.append(prior_vec)
                continuations.append(continuations_table) 

        word_by_word = utt_df.loc[utt_df.partition == 'success']
        word_by_word['prior_vec' ] = prior_vecs
        word_by_word['continuations'] = continuations        

    else: 

        # just get the probability distribution for the next word (no marginalization)
        word_by_word = srilm_word_by_word_prob(utt_df, order, ngram_lm, True, False, vocab, 'success',
            print_debug=False)
    
    prior_vecs_for_success =  np.vstack(word_by_word['prior_vec'])
    continuations = word_by_word['continuations']

    # iterating over the token in word_by_word, get the rank and the probability from the continutions
    score_store = []
    for word_dict in word_by_word.to_dict('records'):
        target_word = word_dict['token']
        prior_rank = word_dict['continuations'].word.to_list().index(target_word)
        prior_prob = word_dict['continuations'].loc[word_dict['continuations'].word == target_word].probability
        entropy = scipy.stats.entropy(word_dict['prior_vec'], base=2)  

        score_df = pd.DataFrame({
            'prior_rank':prior_rank, 
            'prior_prob': prior_prob,
            'token': target_word,
            'entropy':entropy, 
            'num_tokens_in_context':np.nan,
            'bert_token_id': word_dict['bert_token_id']})
        
        score_store.append(score_df)

    scores = pd.concat(score_store)

    return(prior_vecs_for_success, continuations, scores)


def augment_with_ipa(utterances, tokens_with_phono,tokenizer, field):
    '''
        Get a vector of ipa forms corresponding to the length of utterances. If more than one token piece, then jump ahead by the number of token pieces

        Args:
        utterances: utterance form to feed to the model, to be segmented by BERT 
        tokens_with_phono: tokens from PhonBank with ipa forms
        tokenizer: BERT tokenizer
        field: column in tokens_with_phono to return from the map, e.g., actual_phonology or model_phonology
        
        Return:
        mapped_ipa: a vector of tokenized forms from the utterance augmented with phonetic forms from tokens_with_phono as specified by field 
    '''         
    tokens_with_phono['tokens'] = [tokenizer.tokenize(x) for x in tokens_with_phono.gloss]
    
    mapped_ipa = list(np.repeat('', utterances.shape[0]))
    i = 1
    for x in tokens_with_phono.to_dict('record'):
        token_pieces = x['tokens']
        if len(token_pieces) > 1:
            i += len(token_pieces)
        else:
            try:
                mapped_ipa[i] = x[field]
                i+= 1
            except:
                import pdb
                pdb.set_trace()
    return(mapped_ipa)

def find_in_vocab(x, initial_vocab): 
    '''
        Return the label for an index in the vocab

        Args:
        x: index
        initial_vocab: vocabulary of word types in the form of a list
    '''    
    try:
        return(initial_vocab.index(x))
    except:
        return(np.nan)

def get_posteriors(prior_data, levdists, initial_vocab, bert_token_ids=None, scaling_value=None, examples_mode = False):
    '''
        Get the posterior probability of candidate words by combining the priors with a likelihood dependent on levenshtein distances and a free parameter `scaling_value`

        Args:
        prior_data: prior data of the format put out by  compare_successes_failures
        levdists: a matrix of levenshtein distance. for n target words, n,m is the distance of the nth form to the mth word in the initial vocab
        initial_vocab: natural language vocabulary
        bert_token_ids: set of bert_token_ids to limit the contents of prior_data and levdists. This handles the fact that a few utterances are not retrievalbe through BERT but 
        are retrievable through the unigram model query (it is necessary to exclude such forms). So this is None for BERT models, and then defined for subsequent models
        scaling_value: free parameter in the likelihood; see the paper. Higher values assign lower probabilities to larger edit distances
        examples_mode: Whether or not to maintain data related to highest probability words -- used for Examples notebooks, disabled for memory savings.
    '''

    if scaling_value is None: assert False
   
    if bert_token_ids is not None:
        btis = set(bert_token_ids)   
        prior_data['scores']['keep'] = prior_data['scores']['bert_token_id'].isin(btis)
        include_mask = np.argwhere(prior_data['scores']['keep'].to_numpy())[:,0]        
        prior_data['scores'] =  prior_data['scores'].loc[prior_data['scores'].keep]
        prior_data['priors'] =  prior_data['priors'][ include_mask, :]
        levdists = levdists[ include_mask, :]
        
        # correction for unigram models, which have priors for additional tokens
        # that are excluded deep in the bowels of the BERT retrieval methods. 
        # need to subset to bert_token_ods found by other models        
        # also need to limit the scores in some way


    likelihoods = np.exp(-1*scaling_value*levdists)
    unnormalized = np.multiply(prior_data['priors'], likelihoods)
    
    row_sums = np.sum(unnormalized,1)
    
    normalized =  (unnormalized / row_sums[:, np.newaxis])    

    print('Getting posteriors')    
    
    # add entropies
    posterior_entropies = np.apply_along_axis(scipy.stats.entropy, 1, normalized) 
    prior_data['scores']['posterior_entropy'] = posterior_entropies

    def find_word_in_vocab(x):
        location = np.argwhere(initial_vocab == x)
        if len(location) == 1:
            return(location[0])
        else:
            return(np.nan) # token is not present in vocab, cannot rank

    if 'token' in prior_data['scores'].columns: # posterior rank is only defined for communicative successes
        token_ids = np.array([find_word_in_vocab(x) for x in prior_data['scores']['token']]).flatten()
    
        def get_posterior_word_ranks(prob_vec):
            return(np.argsort(prob_vec)[::-1])
        posterior_ranks = np.apply_along_axis(get_posterior_word_ranks, 1, normalized) 

        
        posterior_rank = np.zeros(posterior_ranks.shape[0])
        for i in range(posterior_ranks.shape[0]):
            if np.isnan(token_ids[i]):
                posterior_rank[i] = np.nan
            else:
                posterior_rank[i] = np.argwhere(posterior_ranks[i,:] == token_ids[i])            

        # posterior rank is an operation on normalized
        # each position in  posterior_word_ranks indicates the word's rank
        prior_data['scores']['posterior_rank'] = posterior_rank    
    else:
        prior_data['scores']['posterior_rank'] = np.nan    
    
    prior_entropies = np.apply_along_axis(scipy.stats.entropy, 1, prior_data['priors']) 
    prior_data['scores']['prior_entropy'] = prior_entropies
    # get posterior probability of the correct item into the scores

    # for succesesses, get the word's position in the vocabulary 
    # and add the posterior probability and levenshtein distance
    initial_vocab = list(initial_vocab)
    try:
        prior_data['scores']['position_in_mask'] = [find_in_vocab(x, initial_vocab) if type(x) is str else np.nan for x in prior_data['scores'].token]
    except:
        prior_data['scores']['position_in_mask'] = np.nan #communicative failure
    prior_data['scores']['kl_flat_to_prior'] = np.nan
    prior_data['scores']['kl_flat_to_posterior'] = np.nan
    prior_data['scores']['posterior_probability'] = np.nan
    prior_data['scores']['prior_probability'] = np.nan
    prior_data['scores']['edit_distance'] = np.nan

    for x in ['highest_posterior_words', 'highest_prior_words', 'highest_posterior_probabilities',
        'highest_prior_probabilities']:
            prior_data['scores'][x] = np.nan
            prior_data['scores'][x] = prior_data['scores'][x].astype(object) 


    prior_data['scores']['sample_index'] = range(prior_data['scores'].shape[0])
    prior_data['scores'].set_index('sample_index')
    
    flat_prior = np.repeat(1/len(initial_vocab), len(initial_vocab))
    # Compare all of the distributions to the flat prior cmu_2syl set.

    for i in range(prior_data['scores'].shape[0]):
        if np.isnan(prior_data['scores'].iloc[i]['position_in_mask']):
            pass # initialized as nan        
        else:
            prior_data['scores'].loc[prior_data['scores'].sample_index == i,
                'posterior_probability'] = normalized[i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'edit_distance'] = levdists[i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'prior_probability'] = prior_data['priors'][i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            try:
                prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                    'kl_flat_to_prior'] = scipy.stats.entropy(flat_prior, prior_data['priors'][i,:])
                prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                    'kl_flat_to_posterior'] = scipy.stats.entropy(flat_prior, normalized[i,:])
            except:
                import pdb
                pdb.set_trace()
 
        # get the highest prior probability words + probs
        
        if examples_mode:
            
            num_highest_to_keep = 10 
            highest_prior_indices = np.argsort(prior_data['priors'][i, :])[::-1]
            highest_prior_words = np.array(initial_vocab)[highest_prior_indices][0:num_highest_to_keep]

            prior_data['scores'].loc[prior_data['scores'].sample_index == i,  'highest_prior_words'] = ' ' .join(highest_prior_words)
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                   'highest_prior_probabilities'] = ' '.join([str(x) for x in prior_data['priors'][i, highest_prior_indices]])

            # get the highest poseterior probability words + probs                
            highest_posterior_indices = np.argsort(normalized[i, :])[::-1]
            highest_posterior_words = np.array(initial_vocab)[highest_posterior_indices][0:num_highest_to_keep]

            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                   'highest_posterior_words'] = ' '.join(highest_posterior_words)
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                   'highest_posterior_probabilities'] = ' '.join([str(x) for x in normalized[i, highest_posterior_indices]])
    
    return(prior_data)
