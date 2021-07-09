
# 7/6/21: Parser structure taken from here: https://github.com/NVlabs/SPADE/tree/master/options

import argparse

# 7/6/21: https://docs.python.org/3/library/argparse.html


def split_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('split', type = str, help = 'Which split to use. childes: {all, age}. All others use "all" split.')
    parser.add_argument('dataset', type = str, help = 'Which sub-split to use. childes/all, any other models: {all}. childes/age: {old, young}')
    parser.add_argument('context_width', type = int, default = -1, help = "Context width to use.")
     
    # 7/9/21: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # If you specify False without the type switch below it will still evaluate to True.
     
    parser.add_argument('use_tags', default = False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to include speaker tags. This should only be used as True with the CHILDES models")
    parser.add_argument('model_type', type = str, help = "What model type to use. {'childes' = CHILDES finetuned, 'adult' = BERT off-the-shelf, 'data_unigram' = unigram with CHILDES counts, 'flat_unigram' = completely flat prior}")

    return parser 

# def check_args(raw_args):
        
#     print('here')
#     print(raw_args)
    
#     default_args = vars(raw_args)

#     # Check for valid training inputs.

#     split = default_args['split']
#     dataset = default_args['dataset']
#     model_type = default_args['model_type']

#     is_all_all = (split == 'all') and (dataset == 'all')
#     non_childes_on_non_all = model_type != 'childes' and (not is_all_all)


#     assert (not non_childes_on_non_all), "Only CHILDES finetuned model can be run on non-CHILDES dataset."
#     if split == 'all':
#         assert dataset == 'all', "Trying to use 'all' split with non-all dataset."
#     if split == 'age':
#         assert dataset in {'young', 'old'}, "Trying to use 'age' split with non-age related dataset."
#     if default_args['use_tags']:
#         assert model_types == 'childes', "Trying to use tags with non-childes dataset. The code was not written to support this in mind."
