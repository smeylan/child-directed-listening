import sys
from joblib import Parallel, delayed
import os
import copy
import pandas as pd
import numpy as np
import time
import Levenshtein


sys.path.append("/usr/local/lib/python3.8/site-packages") #for pywrapfst
sys.path.append('/usr/local/lib/python3.6/site-packages')
import pywrapfst

from src.utils import configuration
config = configuration.Config()

def vectorized_compute_all_likelihoods_for_w_over_paths(d_fsa, w_fsas, ws):    
    '''return a vector with entries corresponding to the total path weights from this d_fsa to each word in ws'''
    return([get_likelihood_for_fsas_over_paths(d_fsa, w_fsas, w) for w in ws])

def compute_all_likelihoods_for_w_over_paths_one(list_of_tuples):
    '''wrapper to compute likelihoods for a list of n (d, d_fsa, w_fsa, w, cache_path) tuples. Check if d.npy is in cache_path first'''

    # if cache_path + d exists, read that in and return that   
    distances = []

    for x in list_of_tuples:    
        distances_cache_path = os.path.join(x[4], x[3]+'.npy') 

        if os.path.exists(distances_cache_path):            
            dist_matrix = np.load(distances_cache_path)              
        else:
            dist_matrix = vectorized_compute_all_likelihoods_for_w_over_paths(x[0], x[1], x[2])
            np.save(distances_cache_path, dist_matrix) # write it out in numpy
        
        distances.append(dist_matrix)
    return(np.vstack(distances))

def get_weight_for_path(arc, shortest_paths): 
    '''get the weight of a single arc by iterating in Python'''
    #print(arc)
    path_weight = float(arc.weight)
    finished = False
    while not finished:
        #print(arc.nextstate)
        outgoing_arcs = [x for x in shortest_paths.arcs(arc.nextstate)]
        if len(outgoing_arcs) == 1: 
            arc = outgoing_arcs[0]
            path_weight += float(arc.weight)
        else:
            finished = True

    return(path_weight)

def get_weights_for_paths(shortest_paths):
    '''get the weights for a selection of paths'''
    initial_arcs = [x for x in shortest_paths.arcs(0)]
    return([get_weight_for_path(arc, shortest_paths) for arc in initial_arcs])

def get_likelihood_for_fsas_over_paths(d_fsa, w_fsas, w, num_paths=10, return_type = "probability"):
    '''get the weight of a single arc by iterating in Python'''
    if num_paths <= 0:
        raise ValueError('num_paths must be a positive integer')
        
    w_fsa = w_fsas[w]    
    dw_composed = pywrapfst.compose(w_fsa, d_fsa)
    dw_composed.arcsort(sort_type="ilabel")
       
    if num_paths > 1:
        shortest_paths = pywrapfst.epsnormalize(pywrapfst.shortestpath(dw_composed, nshortest=num_paths))
        if return_type == "shortest_paths": 
            return(shortest_paths)
        if shortest_paths.num_states() > 0:
        
            # take the reverse distance because with multiple shortest paths, 0 is the start state, 1 is the final state
            shortest_distance = pywrapfst.shortestdistance(shortest_paths, reverse=True)
            
            # iterate over all outgoing arcs from the start state  
            path_weights = get_weights_for_paths(shortest_paths)                                
            if return_type == "path_weights":
                return(path_weights)
            shortest_paths_sum = np.sum(np.exp(-1. * np.array(path_weights)))                    
            if return_type == "probability":
                return(shortest_paths_sum)
        else:
            # this is the case where there is no way to compose the d_fsa and the w_fsa
            return(10 ** -20)

    else:
        shortest_path = pywrapfst.shortestpath(dw_composed)
        if shortest_path.num_states() > 0:
            shortest_distance = pywrapfst.shortestdistance(shortest_path)
            return(np.exp(-1 *float(shortest_distance[0])))
        else:
            return(10 ** -20)
        
def string_to_fsa(input_string, sym):
    '''build an FSA for a given input string using the symbol table, sym'''
    
    # first make sure all chars can be converted
    input_list = list(input_string)
    for i in input_list:
        if sym.find(i) == -1:
            raise ValueError('Input character not found')
    
    # build the FSA
    
    f = pywrapfst.VectorFst()
    one = pywrapfst.Weight.one(f.weight_type())
    f.set_input_symbols(sym)
    f.set_output_symbols(sym)
    s = f.add_state()
    f.set_start(s)
    for i in input_list:    
        n = f.add_state()
        f.add_arc(s, pywrapfst.Arc(sym.find(i),
            sym.find(i),  one, n))
        s = n 
    f.set_final(n, 1)
        
    # verify
    if not f.verify():
        raise ValueError('FSA failed to verify')
    return(f)

def write_out_edited_fst(edited_fst, output_path):
    '''writes out a pandas data frame to an FST formatted text file that can them be compiled with OpenFST'''

    # needs to write each state terminal separately
    
    # get the indices of the terminals
    state_weight = np.hstack([np.array([-1]), np.where(edited_fst[[3]] == '')[0]])
    
    first = True
    for i in range(len(state_weight) -1):
        section_start = state_weight[i] + 1
        section_end = state_weight[i+1]         
        #print('Main section: '+str(section_start)+ ' - ', str(section_end))
        
        terminal_start =  state_weight[i+1]
        terminal_end = state_weight[i+1] + 1
        #print('Terminal section: '+str(terminal_start)+ ' - ', str(terminal_end))
        
        ats_section = edited_fst[section_start:section_end]        
        for j in range(3):
            ats_section[[j]] = ats_section[[j]].astype('int')
        #print(ats_section)
            
        if first: 
            ats_section.to_csv(output_path, index=False, header=None, sep='\t')
            first = False
        else:
            ats_section.to_csv(output_path, mode='a', index=False, header=None, sep='\t')

        ats_end = edited_fst.iloc[terminal_start : terminal_end]

        ats_end[0,2] = ''
        ats_end[0,3] = ''
        ats_end[0,4] = ''
        
        ats_end.to_csv(output_path, mode='a',index=False, header=None, sep='\t')

    # catch any remaining arcs
    ats_section = edited_fst[terminal_end:edited_fst.shape[0]]
    for j in range(3):
        ats_section[[j]] = ats_section[[j]].astype('int')
    
    ats_section.to_csv(output_path, mode='a',index=False, header=None, sep='\t')


def get_utf8_sym_table(fit_model):
    
    utf8_points = np.unique(fit_model[[2]].append(fit_model[[3]]) )
    utf8_points = utf8_points[~np.isnan(utf8_points)].astype(int)
    utf8_sym = pd.DataFrame({'utf8':[chr(x) for x in utf8_points], 'sym': utf8_points})
    sym_path = os.path.join(config.project_root, 'output/fst/utf8.sym')
    utf8_sym.at[utf8_sym.sym ==0, 'utf8'] = '<epsilon>'
    utf8_sym.at[utf8_sym.sym ==32, 'utf8'] = 'q' #dummy code. Track down these 32s
    utf8_sym.to_csv(sym_path, header = None, index=False, sep='\t')
    # try reading it in right here
    test = pywrapfst.SymbolTable.read_text(sym_path)

    return(sym_path)


def reconcile_symbols(fit_model, path_to_chi_phones_sym):
    '''generate a transducer and symbol set in the same symbol set which includes all inputs and outputs'''
    ints = [int(x) for x in np.unique(fit_model[[2]]) if not np.isnan(x)]        
    input_symbol_table = pd.DataFrame({'symbol':[chr(x) for x in ints], 'int':ints})
    input_symbol_table.at[input_symbol_table.int ==0, 'symbol'] = '<epsilon>'
    #input_symbol_table.to_csv('test_input_phones.sym', sep='\t', header=None, index=False)
    input_cypher = dict(zip(input_symbol_table.int, input_symbol_table.symbol))
    
    
    output_symbol_table = pd.read_csv(os.path.join(config.project_root, path_to_chi_phones_sym), sep='\t', header=None)
    output_symbol_table.columns = ['symbol','int']
    output_cypher = dict(zip(output_symbol_table.int, output_symbol_table.symbol))
    output_cypher
    
    symbols_not_in_output = set(input_symbol_table.symbol).difference(set(output_symbol_table.symbol))
    
    superset_cypher = copy.copy(output_cypher)
    i = len(output_cypher.keys())
    for symbol in symbols_not_in_output:
        superset_cypher[i] = symbol
        i += 1    
    reverse_superset_cypher = dict(zip(superset_cypher.values(),superset_cypher.keys()))

    fit_model_superset = copy.copy(fit_model)

    # recode the input symbols
    fit_model_superset[[2]] = [reverse_superset_cypher[input_cypher[int(x)]] if not np.isnan(x)
            else '' for x in fit_model[[2]].values[:,0]]

    # recode the output symbols
    fit_model_superset[[3]] = [reverse_superset_cypher[output_cypher[int(x)]] if not np.isnan(x)
            else '' for x in fit_model[[3]].values[:,0]]
    
    fit_model_labeled = copy.copy(fit_model)

    write_out_edited_fst(fit_model_superset, os.path.join(config.project_root, 'output/fst/chi_edited_fst.csv'))

    superset_chi = pd.DataFrame({'sym': reverse_superset_cypher.keys(),
        'utf8':reverse_superset_cypher.values()})
    superset_chi.to_csv(os.path.join(config.project_root, 'output/fst/superset_chi.sym'), header = None, index=False, sep='\t')
    return(fit_model_superset, superset_chi)

def normalize_log_probs(vec):
    vec = vec.values.flatten()
    ps = np.exp(-1 * vec)
    total = np.sum(ps)
    return(ps / total)

def normalize_partition(x): 
    '''for a given selection of FST arcs, for example all where input is a particular symbol, normalize the log probs'''
    df = x[1]
    df[[4]] = -1 * np.log(normalize_log_probs(df[[4]]))
    return(df)

def split(a, n):
    '''split a list into n approximately equal length sublists, appropriate for parallelization'''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_edit_distance_matrix(all_tokens_phono, prior_data,  cmu_2syl_inchildes):    
    '''
    Get an edit distance matrix for matrix-based computation of the posterior.

    all_tokens_phono: corpus in tokenized from, with phonological transcriptions
    prior_data: priors of the form output by `compare_successes_failures_*`    
    cmu_2syl_inchildes: cmu pronunctiations, must have 'word' and 'ipa_short' columns 

    returns: a matrix where each row is an input string from prior_data and each column is a different pronunciation in cmu_2syl_inchildes.
    thus a word type may correspond to multiple columns, and must be reduced using the wfst.reduce_duplicates function
    '''

    print('Getting the Levenshtein distance matrix')

    bert_token_ids = prior_data['scores']['bert_token_id']
    ipa = pd.DataFrame({'bert_token_id':bert_token_ids}).merge(all_tokens_phono[['bert_token_id',
        'actual_phonology_no_dia']])


    levdists = np.vstack([np.array([Levenshtein.distance(target,x) for x in cmu_2syl_inchildes.ipa_short
    ]) for target in ipa.actual_phonology_no_dia]) 
    return(levdists)


def get_wfst_distance_matrix(all_tokens_phono, prior_data, initial_vocab,  cmu_2syl_inchildes, 
    path_to_baum_welch_transducer, path_to_chi_phones_sym, num_cores=24):    
    '''
    Get wfst distance matrix for matrix-based computation of the posterior

    all_tokens_phono: corpus in tokenized from, with phonological transcriptions
    prior_data: priors of the form output by `compare_successes_failures_*`
    initial_vocab: word types corresponding to the softmask mask
    cmu_2syl_inchildes: cmu pronunctiations, must have 'word' and 'ipa_short' columns 
    path_to_baum_welch_transducer: path to the OpenFST transducer yielded by the BaumWelch package
    '''
    bert_token_ids = prior_data['scores']['bert_token_id']
    ipa = pd.DataFrame({'bert_token_id':bert_token_ids}).merge(all_tokens_phono[['bert_token_id',
        'actual_phonology_no_dia']])

    iv = cmu_2syl_inchildes
    
    # [X] Load the transducer, create a covering symbol set, and change the transducer to the data symbol set
    fit_model_superset = pd.read_csv(path_to_baum_welch_transducer, sep='\t', header=None)
    
    utf8_sym_path = get_utf8_sym_table(fit_model_superset)
    utf8_sym = pywrapfst.SymbolTable.read_text(utf8_sym_path)

    #fit_model_superset, superset_chi = reconcile_symbols(fit_model, path_to_chi_phones_sym)
    #superset_chi_sym = pywrapfst.SymbolTable.read_text(os.path.join(config.project_root, 'output/fst/superset_chi.sym'))

    # [X] Change from a joint model to a conditional model.
    # as of 11/10/21, only works for the unigram case
    grouped = list(fit_model_superset.iloc[0:fit_model_superset.shape[0] - 1].groupby(2))
    conditioned = pd.concat([normalize_partition(x) for x in grouped ])
    conditioned[[1]] = [int(x) if not np.isnan(x) else '' for x in conditioned[1]]
    conditioned[[2]] = [int(x) if not np.isnan(x) else '' for x in conditioned[2]]
    conditioned[[3]] = [int(x) if not np.isnan(x) else '' for x in conditioned[3]]

    tail = fit_model_superset.tail(1)
    tail[[1]] = -1 * np.log(1)
    tail[[2]] = [int(x) if not np.isnan(x) else '' for x in tail[2]]
    tail[[3]] = [int(x) if not np.isnan(x) else '' for x in tail[3]]
    tail[[4]] = [int(x) if not np.isnan(x) else '' for x in tail[4]]

    conditioned = pd.concat([conditioned, tail])
    write_out_edited_fst(conditioned, os.path.join(config.project_root, 'output/fst/chi_conditioned_fst.csv'))
    
    chi_conditioned_path = os.path.join(config.project_root, 'output/fst/chi_conditioned.fst')

    os.system('fstcompile --arc_type=standard output/fst/chi_conditioned_fst.csv '+chi_conditioned_path)    
    transducer = pywrapfst.Fst.read(os.path.join(config.project_root, "output/fst/chi_conditioned.fst"))
            
    #[X] translate all words in the vocab into FSAs (w_fsas)and compose with the n-gram transducer
    
    w_fsas = {}
    ws = []
    for w in iv.to_dict('records'):    
        w_fsa = string_to_fsa(w['ipa_short'], utf8_sym)    
        w_in = pywrapfst.compose(w_fsa.arcsort(sort_type="ilabel"), transducer.arcsort(sort_type="ilabel"))
        w_fsas[w['ipa_short']] = w_in.arcsort(sort_type="ilabel")
        ws.append(w['ipa_short'])

    fst_cache_path = os.path.join(config.project_root, config.fst_cache_path)
    if not os.path.exists(fst_cache_path):
        os.mkdir(fst_cache_path)
        
    #[X] translate all observed words (data) into FSAs (d_fsas)
    serial_inputs = [(string_to_fsa(d, utf8_sym).arcsort(sort_type="olabel"), w_fsas, ws, d, fst_cache_path) for d in ipa.actual_phonology_no_dia]
     
    # make the splits on the dfsas        
    if len(serial_inputs) >= num_cores: #avoid stupid weirdness if we have to deal with empty assignments for the workers
        d_fsa_inputs = list(split(serial_inputs, num_cores))
        distances = Parallel(n_jobs=num_cores, verbose=10)(delayed(compute_all_likelihoods_for_w_over_paths_one)(d_fsa_input) for d_fsa_input in d_fsa_inputs)
    else:
        d_fsa_inputs = [serial_inputs]
        distances = [compute_all_likelihoods_for_w_over_paths_one(d_fsa_input) for d_fsa_input in d_fsa_inputs]
    
    # print('Procesing '+str(len(d_fsas))+' d_fsas')
    # distances = []    
    # for d_fsa in d_fsas:        
    #     distances.append(vectorized_compute_all_likelihoods_for_w_over_paths(d_fsa, w_fsas, ws))
    
    # yield the matrix of distances
    
    # !!! make sure that the ordering of the results is not permuted 
    
    return(np.vstack(distances), ipa)   

def reduce_duplicates(wfst_dists, cmu_2syl_inchildes, initial_vocab, max_or_min, cmu_indices_for_initial_vocab):
    '''
    Take a (d x w) distance matrix that includes multiple pronunciations for the same word as separate columns, and return a distance matrix that takes the highest-probability (or lowest distance) true pronunciation for every observation d.
    `wfst_dists`: matrix that includes multiple pronunciations for the same word as separate columns
    `cmu_2syl_inchildes`: DataFrame with `word` column, where words include duplicates for multiple pronunciations, and words are in the same order corresponding to `wfst_dists`
    
    outputs a matrix `wfst_dists_by_word` where each row corresponds to a production and each column correpsonds to a word in initial_vocab

    '''    
    
    wfst_dists_by_word = np.zeros([wfst_dists.shape[0], len(initial_vocab)])  

    for target_production_index in range(wfst_dists.shape[0]):
        for vocab_index in range(len(initial_vocab)):
        
            #find indices where 
            cmu_2syl_indices = cmu_indices_for_initial_vocab[vocab_index]
            if max_or_min == 'max':
                dist = np.max(wfst_dists[target_production_index,cmu_2syl_indices])
            elif max_or_min == 'min':
                dist = np.min(wfst_dists[target_production_index,cmu_2syl_indices])
        
            wfst_dists_by_word[target_production_index, vocab_index] = dist

    return wfst_dists_by_word