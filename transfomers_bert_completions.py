import torch
import numpy as np
import pandas as pd
import scipy.stats
import copy
import time
from string import punctuation
import Levenshtein

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
  return(probs, word_predictions)
  
def compare_completions(context, bertMaskedLM, tokenizer,
    candidates = None):
  continuations = bert_completions(context, bertMaskedLM, tokenizer)
  if candidates is not None:
    return(continuations.loc[continuations.word.isin(candidates)])
  else:
    return(continuations)

def get_completions_for_mask(utt_df, true_word, bertMaskedLM, tokenizer, softmax_mask) :
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

def get_stats_for_failure(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask,context_width_in_utts = None, use_speaker_labels = False, preserve_errors=False):
    '''dummy function because failures mostly just use get_completions_for_mask'''    
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
        before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > this_seq_utt_id - (context_width_in_utts+1)) &all_tokens.transcript_id == this_transcript_id]

        after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1)) &all_tokens.transcript_id == this_transcript_id]
    
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


def get_stats_for_success(all_tokens, selected_utt_id, bertMaskedLM, tokenizer, softmax_mask, context_width_in_utts=None, use_speaker_labels=False, preserve_errors=False):
    '''replace each token one at a time, sending it through get_completions_for_mask'''
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
        mask_positions = np.argwhere((utt_df['partition'] == 'success').to_numpy())[0]            

    for mask_position in mask_positions: 

        utt_df_local = copy.deepcopy(utt_df)    
        utt_df_local.iloc[mask_position, np.argwhere(utt_df_local.columns == 'token')[0][0]] = ['[MASK]']
        utt_df_local.iloc[mask_position,np.argwhere(utt_df_local.columns == 'token_id')[0][0]] = tokenizer.convert_tokens_to_ids(['[MASK]'])

        #concatenate with preceding and following
        if context_width_in_utts is not None:

            this_transcript_id = utt_df.iloc[0].transcript_id

            this_seq_utt_id = utt_df_local.iloc[0].seq_utt_id
            
            before_utt_df = all_tokens.loc[(all_tokens.seq_utt_id < this_seq_utt_id) & (all_tokens.seq_utt_id > this_seq_utt_id - (context_width_in_utts+1)) & all_tokens.transcript_id == this_transcript_id]
            
            after_utt_df = all_tokens.loc[(all_tokens.seq_utt_id > this_seq_utt_id) & (all_tokens.seq_utt_id < this_seq_utt_id + (context_width_in_utts + 1)) & all_tokens.transcript_id == this_transcript_id]
        
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

        this_priors, this_completions, this_stats = get_completions_for_mask(utt_df_local, 
            utt_df.iloc[mask_position].token, bertMaskedLM, tokenizer, softmax_mask) 
        
        this_stats['mask_position'] = mask_position
        this_stats['token'] = utt_df.iloc[mask_position]['token']
        this_stats['utterance_id'] = selected_utt_id
        
        priors.append(this_priors)
        completions.append(this_completions)        
        stats.append(this_stats)
    return(np.transpose(np.vstack(priors)), completions, pd.concat(stats))

    
def get_softmax_mask(tokenizer, token_list):
    vocab_size = len(tokenizer.get_vocab())
    words = np.array(tokenizer.convert_ids_to_tokens(range(vocab_size)))
    mask = np.ones([vocab_size])
    mask[np.array(['##' in x for x in words])] = 0
    mask[np.array([any(p in x for p in punctuation) for x in words])] = 0
    token_list = set(token_list)
    mask[~np.array([x in token_list for x in words])] = 0
    return(np.argwhere(mask)[:,0], words[np.argwhere(mask)][:,0])

def compare_successes_failures(all_tokens, selected_success_utts, selected_yyy_utts, modelLM, tokenizer, softmax_mask, num_context_utts, use_speaker_labels=True, preserve_errors=True):    
    
    print('Computing failure scores')
    import warnings
    warnings.filterwarnings('ignore')

    failure_priors_store = []
    failure_scores_store = []
    
    for sample_yyy_id in selected_yyy_utts:

        rv = get_stats_for_failure(all_tokens, sample_yyy_id, modelLM, tokenizer, softmax_mask, num_context_utts, use_speaker_labels, preserve_errors) 

        if rv is None:
            pass
        else:
            prior, _, score = rv
            failure_scores_store.append(score)
            failure_priors_store.append(prior)
    
    rdict = {}    
    failure_scores = pd.concat(failure_scores_store)
    failure_scores['set'] = 'failure'    
    failure_priors = np.transpose(np.hstack(failure_priors_store))
    
    print('Computing success scores')
    
    success_priors_store = []
    success_scores_store = []

    for success_id in selected_success_utts:
    
        rv = get_stats_for_success(all_tokens, success_id, modelLM, tokenizer, softmax_mask, num_context_utts, use_speaker_labels, preserve_errors) 
        if rv is None:
            pass
            #!!! may want to pass throug a placeholder here 
        else:
            prior, _, score = rv
            success_scores_store.append(score)
            success_priors_store.append(prior)
    
    
    success_scores = pd.concat(success_scores_store)    
    success_scores['set'] = 'success' 
    success_priors = np.vstack(success_priors_store)

    rdict['scores'] = pd.concat([failure_scores, success_scores])
    
    rdict['priors'] = np.vstack([failure_priors, success_priors])

    warnings.filterwarnings('default')

    return(rdict)

def compare_successes_failures_unigram_model(all_tokens, selected_success_utts, selected_yyy_utts, tokenizer, softmax_mask,  child_counts_path, vocab):
    '''unigram model on CHILDES productions''' 

    
    # child production stats only
    
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
    
    rdict = {}
    rdict['scores'] = pd.concat([failure_scores, success_scores])

    # needs slicing and dicing

    prior_vec = unigram_model['prob'].to_numpy()
    # where is 1017609.0 or 1369403.0

    rdict['priors'] = test = np.vstack([
        np.vstack([prior_vec for x in range(failure_scores.shape[0])]),
        np.vstack([prior_vec for x in range(success_scores.shape[0])])
    ]) 
    # needs a switch to use a flat prior

    return(rdict)


def augment_with_ipa(successes, tokens_with_phono,tokenizer, field):
     
    tokens_with_phono['tokens'] = [tokenizer.tokenize(x) for x in tokens_with_phono.gloss]
    
    mapped_ipa = list(np.repeat('', successes.shape[0]))
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
    try:
        return(initial_vocab.index(x))
    except:
        return(np.nan)

def get_posteriors(prior_data, levdists, initial_vocab, bert_token_ids=None):
    
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
        # also need to limit the scores in some way\
        
    likelihoods = np.exp(-1*levdists)
    unnormalized = np.multiply(prior_data['priors'], likelihoods)
    row_sums = np.sum(unnormalized,1)
    normalized =  unnormalized / row_sums[:, np.newaxis]
    
    # add entropies
    entropies = np.apply_along_axis(scipy.stats.entropy, 1, normalized) 
    prior_data['scores']['posterior_entropy'] = entropies
    # get posterior probability of the correct item into the scores

    # for succesesses, get the word's position in the vocabulary 
    # and add the posterior probability and levenshtein distance
    initial_vocab = list(initial_vocab)
    prior_data['scores']['position_in_mask'] = [find_in_vocab(x, initial_vocab) if type(x) \
        is str else np.nan for x in prior_data['scores'].token ]
    prior_data['scores']['kl_flat_to_prior'] = np.nan
    prior_data['scores']['kl_flat_to_posterior'] = np.nan
    prior_data['scores']['posterior_surprisal'] = np.nan
    prior_data['scores']['prior_surprisal'] = np.nan
    prior_data['scores']['edit_distance'] = np.nan
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

    return(prior_data)

def sample_models_across_time(utts_with_ages, all_tokens_phono, models, initial_vocab, cmu_in_initial_vocab, num_samples = 2000):

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
            
                edit_distances_for_age_interval = get_edit_distance_matrix(all_tokens_phono, 
                priors_for_age_interval, cmu_in_initial_vocab)   
                              
            elif model['type'] == 'unigram':
                priors_for_age_interval = compare_successes_failures_unigram_model(
                    all_tokens_phono, selected_success_utts.utterance_id, 
                    selected_yyy_utts.utterance_id, **model['kwargs'])

            edit_distances_for_age_interval = get_edit_distance_matrix(all_tokens_phono, 
                priors_for_age_interval, cmu_in_initial_vocab)            

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

def get_edit_distance_matrix(all_tokens_phono, prior_data,  cmu_2syl_inchildes):    
    bert_token_ids = prior_data['scores']['bert_token_id']
    ipa = pd.DataFrame({'bert_token_id':bert_token_ids}).merge(all_tokens_phono[['bert_token_id',
        'actual_phonology_no_dia']])
    levdists = np.vstack([np.array([Levenshtein.distance(target,x) for x in cmu_2syl_inchildes.ipa_short
    ]) for target in ipa.actual_phonology_no_dia])
    return(levdists)    