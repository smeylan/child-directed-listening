
# 7/6/21: Parser structure taken from here: https://github.com/NVlabs/SPADE/tree/master/options

import argparse

def split_parser():
    
    # 7/6/21: https://docs.python.org/3/library/argparse.html
    # For syntax references
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('split', type = str, help = 'Which split to use. childes: {all, age}. All others use "all" split.')
    parser.add_argument('dataset', type = str, help = 'Which sub-split to use. childes/all, any other models: {all}. childes/age: {old, young}')
    parser.add_argument('context_width', type = int, default = -1, help = "Context width to use.")
     
    # 7/9/21: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # If you specify False without the type switch below it will still evaluate to True.
     
    parser.add_argument('use_tags', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")
    
    # end stackoverflow cite
    
    parser.add_argument('model_type', type = str, help = "What model type to use. {'childes' = CHILDES finetuned, 'adult' = BERT off-the-shelf, 'data_unigram' = unigram with CHILDES counts, 'flat_unigram' = completely flat prior}")

    # end syntax references
    return parser 

# end parser structure taken from cite
