import torch
import numpy as np
import pandas as pd
import scipy.stats
import copy
import time
from string import punctuation
import Levenshtein

def softmax(x, axis=None):
    '''
        Compute softmax on a vector or matrix

        Args:
        x: vector or matrix
        axis: none if a vector, else dimension over which to compute the softmax
    '''
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


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
  
  word_predictions  = pd.DataFrame({'prob': probs, 'word':words})
  word_predictions = word_predictions.sort_values(by='prob', ascending=False)    
  word_predictions['rank'] = range(word_predictions.shape[0])
  return(probs, word_predictions)
  
  
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
        rank = true_completion['rank']
        prob =  true_completion['prob']
    else:
        rank = np.nan
        prob = np.nan    
    
    entropy = scipy.stats.entropy(completions.prob, base=2)
    return(priors, completions , pd.DataFrame({'rank':[rank], 'prob': [prob], 'entropy':[entropy], 'num_tokens_in_context':[utt_df.shape[0]-1],
    'bert_token_id' : utt_df.loc[utt_df.token == '[MASK]'].bert_token_id}))

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
    utt_df = all_tokens.loc[all_tokens.id == selected_utt_id]
        
    if utt_df.shape[0] == 0:
        return None
    else:
        # convert the @ back to a mask for the target word
        utt_df.loc[utt_df.token == 'yyy','token'] = '[MASK]'
        utt_df.loc[utt_df.token == '[MASK]','token_id'] = 103


    if context_width_in_utts is not None:   

        # limit to the current transcript
        this_transcript_id = utt_df.iloc[0].transcript_id

        this_seq_utt_id = utt_df.iloc[0].seq_utt_id
        before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > (this_seq_utt_id - (context_width_in_utts+1))) & (all_tokens['transcript_id'] == this_transcript_id)]

        after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
    
        # ready to use before_utt_df and after_utt_df    
        sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})

        if before_utt_df.shape[0] > 0:
            before_by_sent = [x[1] for x in before_utt_df.groupby(['seq_utt_id'])]    
            i = n = 1
            while i < len(before_by_sent):
                before_by_sent.insert(i, sep_row)
                i += (n+1)
            before_by_sent_df = pd.concat(before_by_sent)
        else:
            before_by_sent_df = pd.DataFrame()

        if after_utt_df.shape[0] > 0:
            after_by_sent = [x[1] for x in after_utt_df.groupby(['seq_utt_id'])]    
            i = n = 1
            while i < len(after_by_sent):
                after_by_sent.insert(i, sep_row)
                i += (n+1)
            after_by_sent_df = pd.concat(after_by_sent)
        else:
            after_by_sent_df = pd.DataFrame()

        utt_df = pd.concat([before_by_sent_df, sep_row, utt_df, sep_row, after_by_sent_df])

        # if preserve_errors:
        #     #convert @ back to yyy for context items
        #     utt_df.loc[utt_df.token == '@','token'] = 'yyy'
        #     utt_df.loc[utt_df.token == 'yyy','token_id'] = tokenizer.convert_tokens_to_ids(['yyy'])[0]
        #     #convert @ back to xxx for context items
        #     utt_df.loc[utt_df.token == '$','token'] = 'xxx'
        #     utt_df.loc[utt_df.token == 'xxx','token_id'] = tokenizer.convert_tokens_to_ids(['xxx'])[0]

        if np.sum(utt_df.token == '[MASK]') > 1:
            print('Multiple masks in the surrounding context')
            import pdb
            pdb.set_trace()        
    
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
    
    utt_df = all_tokens.loc[all_tokens.id ==     selected_utt_id]    
    
    if utt_df.shape[0] == 0:
        return(None)

    if not use_speaker_labels : 
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

            this_seq_utt_id = utt_df_local.iloc[0].seq_utt_id
            
            before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > this_seq_utt_id - (context_width_in_utts+1)) & (all_tokens.transcript_id == this_transcript_id)]
            
            after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1)) & (all_tokens.transcript_id == this_transcript_id)]
        
            # ready to use before_utt_df and after_utt_df    
            sep_row = pd.DataFrame({'token':['[SEP]'], 'token_id':tokenizer.convert_tokens_to_ids(['[SEP]'])})
            if before_utt_df.shape[0] > 0:
                before_by_sent = [x[1] for x in before_utt_df.groupby(['seq_utt_id'])]    
                i = n = 1
                while i < len(before_by_sent):
                    before_by_sent.insert(i, sep_row)
                    i += (n+1)
                before_by_sent_df = pd.concat(before_by_sent)
            else:
                before_by_sent_df = pd.DataFrame()

            if after_utt_df.shape[0] > 0:
                after_by_sent = [x[1] for x in after_utt_df.groupby(['seq_utt_id'])]    
                i = n = 1
                while i < len(after_by_sent):
                    after_by_sent.insert(i, sep_row)
                    i += (n+1)
                after_by_sent_df = pd.concat(after_by_sent)
            else:
                after_by_sent_df = pd.DataFrame()

            utt_df_local = pd.concat([before_by_sent_df, sep_row, utt_df_local, sep_row, after_by_sent_df])

            # if preserve_errors:
            #     #convert @ back to yyy for context items
            #     utt_df_local.loc[utt_df_local.token == '@','token'] = 'yyy'
            #     utt_df_local.loc[utt_df_local.token == 'yyy','token_id'] = tokenizer.convert_tokens_to_ids(['yyy'])[0]
            #     #convert @ back to xxx for context items
            #     utt_df_local.loc[utt_df_local.token == '$','token'] = 'xxx'
            #     utt_df_local.loc[utt_df_local.token == 'xxx','token_id'] = tokenizer.convert_tokens_to_ids(['xxx'])[0]

            #     if np.sum(utt_df_local.token == '[MASK]') > 1:
            #         print('Multiple masks in the surrounding context')
            #         import pdb
            #         pdb.set_trace()
            # else: 
            #     #convert xxx and yyy in the context to masks
            #     utt_df_local.loc[utt_df_local.token == '@','token'] = '[MASK]'
            #     utt_df_local.loc[utt_df_local.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            #     #convert @ back to xxx for context items
            #     utt_df_local.loc[utt_df_local.token == '$','token'] = '[MASK]'
            #     utt_df_local.loc[utt_df_local.token == '[MASK]','token_id'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        if not use_speaker_labels:
            # remove the speaker labels
            utt_df_local = utt_df_local.loc[~utt_df_local.token.isin(['[chi]','[cgv]'])]

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
    mask[np.array(['##' in x for x in words])] = 0
    mask[np.array([any(p in x for p in punctuation) for x in words])] = 0
    token_list = set(token_list)
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
            scores: a datframe of length n containing concatenated entropy scores, ranks, surprisals. Failures are stacked on top of successes and are identified by a bert_token_id
    '''
    
    print('Computing failure scores')
    import warnings
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

def compare_successes_failures_unigram_model(all_tokens, selected_success_utts, selected_yyy_utts, tokenizer, softmax_mask,  child_counts_path, vocab):
    '''
        Get prior probaiblities, completions, and scores from a unigram model (limiting to the vocab) or flat prior for a list of utterance ids for communicative successes and a list of utterance ids for communicative failures 

        Args:
        all_tokens: data frame containing selected_utt_id plus surrounding context
        selected_suuccess_utts: utterance ids known to be communicative successes
        selected_yyy_utts: utterance ids known to be communicative failures        
        tokenizer: BERT tokenizer
        softmax_mask: a vector of indices of vocabulary items over which to compute the softmax
        child_counts_path: file to read with 'word' and 'count' in the column titles to use for unigram counts. If set to None, return a uniform prior
        vocab: a string representation of the vocab

        Returns: dictionary with two keys:            
            priors: n * m matrix of prior probabilities, where n is the number of communicative failures + communicative successes, and m is the size of the vocab identified by the softmax map. Failures are stacked on top of successes and are identified by a bert_token_id
            scores: a datframe of length n containing concatenated entropy scores, ranks, surprisals. Failures are stacked on top of successes and are identified by a bert_token_id
    '''    

    unigram_model = pd.DataFrame({'word':vocab})
    
    if child_counts_path is not None: 
        childes_counts = pd.read_csv(child_counts_path)    
        unigram_model = unigram_model.merge(childes_counts, how='left')
        unigram_model = unigram_model.fillna(0) 
        unigram_model['count'] = unigram_model['count'] + .01 #additive smoothing
        
        unigram_model['prob'] = unigram_model['count'] / np.sum(unigram_model['count'])
    
    else:
        # build a flat prior: assign all words equal probability
        unigram_model['prob'] = 1/unigram_model.shape[0]        

    # for successes, get the probability of all words        
    success_utt_contents = all_tokens.loc[all_tokens.id.isin(selected_success_utts)]
    success_utt_contents = success_utt_contents.loc[~success_utt_contents.token.isin(['[chi]'])]

    success_scores = success_utt_contents[['token','bert_token_id']].merge(unigram_model, left_on='token', right_on='word', how='left')
    
    # entropy will be the same for all success and failure tokens
    constant_entropy = scipy.stats.entropy(unigram_model['prob'])
    success_scores['entropy'] = constant_entropy
    success_scores['set'] = 'success'
    
    # need to retrieve the failures in the same way so I can limit by bert_token_id
    failure_scores = all_tokens.loc[(all_tokens.id.isin(selected_yyy_utts)) &
      (all_tokens.token == 'yyy') ][['token','bert_token_id']]

    failure_scores['entropy'] = constant_entropy
    failure_scores['set'] = 'failure'

    prior_vec = unigram_model['prob'].to_numpy()
    # where is 1017609.0 or 1369403.0
    
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

def get_posteriors(prior_data, levdists, initial_vocab, bert_token_ids=None, beta_value = 3.2):
    '''
        Get the posterior probability of candidate words by combining the priors with a likelihood dependent on levenshtein distances and a free parameter beta

        Args:
        prior_data: prior data of the format put out by  compare_successes_failures
        levdists: a matrix of levenshtein distance. for n target words, n,m is the distance of the nth form to the mth word in the initial vocab
        initial_vocab: natural language vocabulary
        bert_token_ids: set of bert_token_ids to limit the contents of prior_data and levdists. This handles the fact that a few utterances are not retrievalbe through BERT but 
        are retrievable through the unigram model query (it is necessary to exclude such forms) 
        beta_value: free parameter in the likelihood; see the paper. Higher values assign lower probabilities to larger edit distances
    '''

    
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

    likelihoods = np.exp(-1*beta_value*levdists)
    unnormalized = np.multiply(prior_data['priors'], likelihoods)
    row_sums = np.sum(unnormalized,1)
    normalized =  unnormalized / row_sums[:, np.newaxis]
    
    # add entropies
    posterior_entropies = np.apply_along_axis(scipy.stats.entropy, 1, normalized) 
    prior_data['scores']['posterior_entropy'] = posterior_entropies
    
    
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
    prior_data['scores']['posterior_surprisal'] = np.nan
    prior_data['scores']['prior_surprisal'] = np.nan
    prior_data['scores']['edit_distance'] = np.nan

    for x in ['highest_posterior_words', 'highest_prior_words', 'highest_posterior_probabilities',
        'highest_prior_probabilities']:
            prior_data['scores'][x] = np.nan
            prior_data['scores'][x] = prior_data['scores'][x].astype(object) 


    prior_data['scores']['sample_index'] = range(prior_data['scores'].shape[0])
    prior_data['scores'].set_index('sample_index')
    
    flat_prior = np.repeat(1/len(initial_vocab), len(initial_vocab))

    for i in range(prior_data['scores'].shape[0]):
        if np.isnan(prior_data['scores'].iloc[i]['position_in_mask']):
            pass # initialized as nan        
        else:
            prior_data['scores'].loc[prior_data['scores'].sample_index == i,
                'posterior_surprisal'] = normalized[i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'edit_distance'] = levdists[i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'prior_surprisal'] = prior_data['priors'][i, \
                int(prior_data['scores'].iloc[i]['position_in_mask'])]
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'kl_flat_to_prior'] = scipy.stats.entropy(flat_prior, prior_data['priors'][i,:])
            prior_data['scores'].loc[prior_data['scores'].sample_index == i, 
                'kl_flat_to_posterior'] = scipy.stats.entropy(flat_prior, normalized[i,:])


        # get the highest prior probability words + probs
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

def sample_models_across_time(utts_with_ages, all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab, num_samples = 1000):
    '''
    Top-level method to sample all models for a set of communicative successes and failures split across different ages

        Args:
        utts_with_ages: utterances with ages associated (sampling occurs within the function)
        all_tokens_phono: corpus in tokenized from, with phonological transcriptions
        models: list of models and their associated tokenizers, softmax masks, and parameters
        initial_vocab: word types corresponding to the softmask mask !!! potentially brittle:  different softmax masks per model 
        cmu_in_initial_vocab: cmu pronunciations for the initial vocabulary !!! potentially brittle interaction with initial voca

        Returns:
        scores_across_time: a dataframe of scores for each word for each model for each time period
    '''


    score_store = []
    for age in np.unique(utts_with_ages.year):
        print('Running models for age '+str(age))
        selected_success_utts = utts_with_ages.loc[(utts_with_ages.set == 'success') 
            & (utts_with_ages.year == age)]
        if selected_success_utts.shape[0] > num_samples:
            selected_success_utts = selected_success_utts.sample(num_samples, replace=False)

        selected_yyy_utts = utts_with_ages.loc[(utts_with_ages.set == 'failure') 
            & (utts_with_ages.year == age)]
        if selected_yyy_utts.shape[0] > num_samples:
            selected_yyy_utts = selected_yyy_utts.sample(num_samples, replace=False)

        #print(selected_success_utts.utterance_id[0:10])
        #print(selected_yyy_utts.utterance_id[0:10])
        for model in models:
            print('Running model '+model['title']+'...')
            if model['type'] == 'BERT':
                priors_for_age_interval = compare_successes_failures(
                    all_tokens_phono, selected_success_utts.utterance_id, 
                    selected_yyy_utts.utterance_id, **model['kwargs'])
                              
            elif model['type'] == 'unigram':
                priors_for_age_interval = compare_successes_failures_unigram_model(
                    all_tokens_phono, selected_success_utts.utterance_id, 
                    selected_yyy_utts.utterance_id, **model['kwargs'])

            edit_distances_for_age_interval = get_edit_distance_matrix(all_tokens_phono, 
                priors_for_age_interval, initial_vocab, cmu_in_initial_vocab)            

            if model['type'] == 'BERT':
                posteriors_for_age_interval = get_posteriors(priors_for_age_interval, 
                    edit_distances_for_age_interval, initial_vocab)
            elif model['type'] == 'unigram':
                # special unigram hack
                posteriors_for_age_interval = get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, initial_vocab, score_store[-1].bert_token_id)
            

            posteriors_for_age_interval['scores']['model'] = model['title']
            posteriors_for_age_interval['scores']['age'] = age
            score_store.append(copy.deepcopy(posteriors_for_age_interval['scores']))
        
    scores_across_time = pd.concat(score_store)
    return(scores_across_time)

def get_edit_distance_matrix(all_tokens_phono, prior_data, initial_vocab,  cmu_2syl_inchildes):    
    '''
    Get an edit distance matrix for matrix-based computation of the prior

    all_tokens_phono: corpus in tokenized from, with phonological transcriptions
    prior_data: priors of the form output by `compare_successes_failures_*`
    initial_vocab: word types corresponding to the softmask mask
    cmu_2syl_inchildes: cmu pronunctiations, must have 'word' and 'ipa_short' columns 
    '''

    bert_token_ids = prior_data['scores']['bert_token_id']
    ipa = pd.DataFrame({'bert_token_id':bert_token_ids}).merge(all_tokens_phono[['bert_token_id',
        'actual_phonology_no_dia']])

    iv = pd.DataFrame({'word':initial_vocab})
    iv = iv.merge(cmu_2syl_inchildes, how='left')

    levdists = np.vstack([np.array([Levenshtein.distance(target,x) for x in iv.ipa_short
    ]) for target in ipa.actual_phonology_no_dia])    
    return(levdists)    



def sample_across_models(utterance_ids, success_utts, yyy_utts, all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab, beta_values):
    '''
        Top-level method to sample all models for a set of communicative successes and failures. Allows for adjusting the beta value

        Args:
        utterance_ids: set of utterance ids to retrieve
        success_utts: set of communicative success utterances 
        yyy_utts: set of communicative failure utterances 
        all_tokens_phono: corpus in tokenized from, with phonological transcriptions
        models: list of models and their associated tokenizers, softmax masks, and parameters
        initial_vocab: word types corresponding to the softmask mask !!! potentially brittle:  different softmax masks per model 
        cmu_in_initial_vocab: cmu pronunciations for the initial vocabulary !!! potentially brittle interaction with initial vocab       
        beta values: list of beta values over which to iterate

        Returns:
        scores_across_models: a dataframe of scores for each word for each model for each beta

    '''

    score_store = []
    for model in models:
        print('Running model '+model['title']+'...')

        selected_success_utts =   success_utts.loc[success_utts.utterance_id.isin(utterance_ids)]
        selected_yyy_utts =  yyy_utts.loc[yyy_utts.utterance_id.isin(utterance_ids)]        

        # get the priors
        if model['type'] == 'BERT':
            priors_for_age_interval = compare_successes_failures(
                all_tokens_phono, selected_success_utts.utterance_id, 
                selected_yyy_utts.utterance_id, **model['kwargs'])
                            
        elif model['type'] == 'unigram':
            priors_for_age_interval = compare_successes_failures_unigram_model(
                all_tokens_phono, selected_success_utts.utterance_id, 
                selected_yyy_utts.utterance_id, **model['kwargs'])

        edit_distances_for_age_interval = get_edit_distance_matrix(all_tokens_phono, 
            priors_for_age_interval, initial_vocab, cmu_in_initial_vocab)            

        for beta_value in beta_values:

            # get the posteriors        
            if model['type'] == 'BERT':
                posteriors_for_age_interval = get_posteriors(priors_for_age_interval, 
                    edit_distances_for_age_interval, initial_vocab, None, beta_value)
            
            elif model['type'] == 'unigram':
                # special unigram hack
                posteriors_for_age_interval = get_posteriors(priors_for_age_interval, edit_distances_for_age_interval, 
                    initial_vocab, score_store[-1].bert_token_id, beta_value)
            
            posteriors_for_age_interval['scores']['beta_value'] = beta_value
            posteriors_for_age_interval['scores']['model'] = model['title']
            score_store.append(copy.deepcopy(posteriors_for_age_interval['scores']))
    
    scores_across_models = pd.concat(score_store)
    return(scores_across_models)
