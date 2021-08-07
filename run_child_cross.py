
from utils_child import child_parser, child_models, utils_child
from utils import split_gen, load_models

from datetime import datetime

if __name__ == '__main__':
    
    start_time = str(datetime.today())
    parser = child_parser.child_parser()
    
    # 7/7/21: https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments    
    raw_args = parser.parse_known_args()[0]
    # end cites
    # Not sure why known args is necessary here.
    
    this_model_args = vars(raw_args)
    data_child = this_model_args['data_child']
    prior_child = this_model_args['prior_child']
   
    this_model_dict = child_models.get_child_model_dict(prior_child)
    
    scores, beta_used = utils_child.score_cross_prior(data_child, prior_child)
    
    score_path = utils_child.get_cross_path(data_child, prior_child)
    
    scores.to_pickle(score_path)
    
    print(f'Computations complete for: {data_child}, {prior_child}')
    print(f'Scores saved to: {score_path}')
    
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
   