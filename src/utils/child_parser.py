
# 7/6/21: Parser structure taken from here: https://github.com/NVlabs/SPADE/tree/master/options

import argparse

def child_parser():
    
    # 7/6/21: https://docs.python.org/3/library/argparse.html
    # For syntax references
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_child', type = str, help = 'Which child corpus to use.')
    parser.add_argument('prior_child', type = str, help = 'Which prior to use.')
    
    # end syntax references
    return parser 

# end parser structure taken from cite
