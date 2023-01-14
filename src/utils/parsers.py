
# 7/6/21: Parser structure taken from here: https://github.com/NVlabs/SPADE/tree/master/options

import argparse

def split_parser():
    
    # 7/6/21: https://docs.python.org/3/library/argparse.html
    # For syntax references
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type = str, help = 'The name of the task name for groups of scripts to be run together on the cluster')


    parser.add_argument('--task_phase', type = str, help = 'Which phase this corresponds to sample, extract_data, train, fit, or eval')
        
    parser.add_argument('--test_split', type = str, help = 'Which split to use. childes: {all, age}. All others use "all" split.')
    
    parser.add_argument('--test_dataset', type = str, help = 'Which sub-split to use. childes/all, any other models: {all}. childes/age: {old, young}')

    parser.add_argument('--context_width', default = None, help = "Context width to use.")    

    parser.add_argument('--use_tags', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")

    parser.add_argument('--model_type', type = str, help = "What model type to use. {'childes' = CHILDES finetuned, 'adult' = BERT off-the-shelf, 'data_unigram' = unigram with CHILDES counts, 'flat_unigram' = completely flat prior}")

    parser.add_argument('--training_split', type = str, default = None, help = "Training split, in case it is different from `split`. If only `split` is defined, then the code will use whatever is specified by `split` for BOTH train and test")

    # generally we assume that dataset will determine BOTH the training data for the model and the test data
    # if we want to separate those two things, we can specify "training_dataset" separately 
    parser.add_argument('--training_dataset', type = str, default = None, help = "Training dataset, in case it is different from `dataset`. If only `dataset` is defined, then the code will use whatever is specified by `dataset` for BOTH train and test")

    parser.add_argument('--order', default = None, type=int, help = "Order of the n-gram to use (e.g. 3 = trigram model, conidtioning on two preceding words)")

    parser.add_argument('--contextualized', default = None, help = "compute n-gram probabilities by marginalizing over all continuations. 1 = true")

    parser.add_argument('--ngram_path', type=str, default = None, help = "path to the n-gram model in LM format")

    parser.add_argument('--batch_size', default = 2, type=int, help = "Size of the batch to use in GPT-2 baseed models, default is 2")

    return parser 

# end parser structure taken from cite
