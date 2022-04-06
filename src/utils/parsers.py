
# 7/6/21: Parser structure taken from here: https://github.com/NVlabs/SPADE/tree/master/options

import argparse

def split_parser():
    
    # 7/6/21: https://docs.python.org/3/library/argparse.html
    # For syntax references
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('split', type = str, help = 'Which split to use. childes: {all, age}. All others use "all" split.')
    
    parser.add_argument('training_split', type = str, default = None, help = "Training split, in case it is different from `split`. If only `split` is defined, then the code will use whatever is specified by `split` for BOTH train and test")

    parser.add_argument('dataset', type = str, help = 'Which sub-split to use. childes/all, any other models: {all}. childes/age: {old, young}')

    # generally we assume that dataset will determine BOTH the training data for the model and the test data
    # if we want to separate those two things, we can specify "training_dataset" separately 
    parser.add_argument('training_dataset', type = str, default = None, help = "Training dataset, in case it is different from `dataset`. If only `dataset` is defined, then the code will use whatever is specified by `dataset` for BOTH train and test")

    parser.add_argument('context_width', type = int, default = -1, help = "Context width to use.")
     
    # 7/9/21: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # If you specify False without the type switch below it will still evaluate to True.
     
    parser.add_argument('use_tags', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")
    
    # end stackoverflow cite
    
    parser.add_argument('model_type', type = str, help = "What model type to use. {'childes' = CHILDES finetuned, 'adult' = BERT off-the-shelf, 'data_unigram' = unigram with CHILDES counts, 'flat_unigram' = completely flat prior}")

    # end syntax references
    return parser 

# end parser structure taken from cite
