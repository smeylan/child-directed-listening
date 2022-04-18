from os.path import join
from src.utils import configuration
config = configuration.Config()

def get_directory(
		test_split, 
		training_split,
		test_dataset,
		training_dataset,
		model_type,
		use_tags,
		context_width,        
		task_name,
		task_phase):

	'''single source of path names, using a model specification plus specifying a task phase'''

	tags_str = 'with_tags' if use_tags else 'no_tags'	


	if task_phase == 'data':

		if context_width is not None:
			raise ValueError('context_width must be None in training. Context_width is only used in fitting and evaluation, and re-uses the same trained model')


		if test_split is not None:
			raise ValueError('test_split must be None in training. test_split is only used in fitting and evaluation')

		if test_dataset is not None:
			raise ValueError('test_split must be None in training. test_dataset is only used in fitting and evaluation')			

		path = join(config.model_dir, task_phase, training_split + '_' + training_dataset + '_'  + tags_str) 
	
	elif task_phase == 'train':

		if context_width is not None:
			raise ValueError('context_width must be None in training. Context_width is only used in fitting and evaluation, and re-uses the same trained model')


		if test_split is not None:
			raise ValueError('test_split must be None in training. test_split is only used in fitting and evaluation')

		if test_dataset is not None:
			raise ValueError('test_split must be None in training. test_dataset is only used in fitting and evaluation')		

		path = join(config.model_dir, task_phase, model_type + '_'+ training_split + '_' + training_dataset + '_'  + tags_str)

	elif task_phase in ('fitting','eval'):
		
		path = join(config.model_dir, task_phase, model_type + '_'+ split_name + '_' + dataset_name + '_' + training_dataset_name + '_' + tags_str + '_' + context_width)
	else:
		raise ValueError('Task phase not recognized')


	return(path)


def get_slurm_script_path(
    test_split, 
    training_split,
    test_dataset,
    training_dataset,
    model_type,
    use_tags,
    context_width,        
    task_name,
    task_phase
):

        tags_str = 'with_tags' if use_tags else 'no_tags'    

        if task_phase == 'train':

            path =  f'{task_phase}_{model_type}_{training_split}_{training_dataset}_{context_width}_{tags_str}.sh'
        else:

            raise NotImplementedError

        return(path)


def get_sample_csv_path(task_phase, split, dataset, data_type, age = None):
    
    n = get_n(task_phase)
    
    assert ( (age is None) and (task_phase == 'fitting' or split == 'Providence') ) or ( (age is not None) and (task_phase == 'eval') )
    age_str = f'_{float(age)}' if age is not None else ''
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    # where should the sampling csvs get stored? at samples for a run,
    # dvided by split and then by datase
    sample_folder = os.path.join(
    	config.sample_dir,
    	split, 
    	dataset
    )
    # same samples can be used across task_name
    # split and daset determine the 

    if not exists(sample_folder):
    	os.makedirs(sample_folder)
    
    this_data_path = join(sample_folder, f'{task_phase}_{data_type}_utts_{n}_{age_str}.csv')
    
    return this_data_path