from os.path import join, exists
import os
from src.utils import configuration
config = configuration.Config()


def validate_spec_dict(spec_dict, spec_dict_params):
	''' make sure all necessary keys are specified for the spec_dict'''
	for key in spec_dict_params:
		if key not in spec_dict:
			raise ValueError('spec_dict must contain the key '+key)
	
	for key in spec_dict:
		if key not in spec_dict_params:
			if key not in ['title','kwargs','examples_mode']: # some reseved keys for building model fitting and eval dicts
				raise ValueError('spec_dict has extraneous key '+key)

def validate_phase(phase, known_phases):
	if phase not in known_phases:
		raise ValueError('Phase "'+phase+'" not recognized!')

def confirm_values_are_none(spec_dict, varnames):
	for var in varnames:
		if spec_dict[var] is not None:
			raise ValueError(var+' must be None in '+spec_dict['task_phase'])

def confirm_values_are_not_none(spec_dict, varnames):
	for var in varnames:
		if spec_dict[var] is None:
			raise ValueError(var+' must be defined in '+spec_dict['task_phase'])

def validate_training_params(spec_dict):
	if [spec_dict['training_split'], spec_dict['training_dataset']] not in config.training_datasets:
		raise ValueError("training parameters don't correspond to a known dataset")

def validate_test_params(spec_dict):
	if [spec_dict['test_split'], spec_dict['test_dataset']] not in config.test_datasets:
		raise ValueError("test parameters don't correspond to a known dataset")

def get_directory(spec_dict):	

	''' 
	Single specification for all directory structures in sampling, data extraction, training, fitting, evaluation, and analysis

	Paths follow 
	output/experiments<exp_identifier>/<phase>/<n_samples>/<identifier>
	
	<exp_identifier>: "full_scale"
	<phase>: "sample", "extract_data", "train", "fit", "eval", "analyze"
	<n_samples>:n=x where x is the number of items that are drawn for....
	<identifier>: <training_split>_<training_dataset>(x<tags>)(x<model_type>)(x<test_split>_<test_dataset>_<context_width>)

	for <identifier>:
	when phase == "sample" use the first part;
	when phase == "extract_data" use the second part;
	when phase == 'train', add the model_type 
	when phase == "fit", "eval", "analyze", add the details of the test set and the context width
 
	`task_name` is not currently used, but it could be useful (eg non_child vs. child).
	non_child vs. child is implicit in the training data, etc.
	'''

	config = configuration.Config()

	validate_spec_dict(spec_dict, config.spec_dict_params)	
	validate_phase(spec_dict['task_phase'], config.task_phases)
	validate_training_params(spec_dict)	


	if spec_dict['task_phase'] == 'sample':		

		confirm_values_are_none(spec_dict, ['context_width', 'test_dataset','test_split'])
		confirm_values_are_not_none(spec_dict, ['training_split', 'training_dataset', 'n_samples'])	

		n_str = 'n='+str(spec_dict['n_samples'])

		path = join(config.exp_dir, spec_dict['task_phase'], n_str, spec_dict['training_split'] + '_' + spec_dict['training_dataset']) 


	elif spec_dict['task_phase'] == 'extract_data':		

		confirm_values_are_none(spec_dict, ['context_width', 'test_dataset','test_split'])
		confirm_values_are_not_none(spec_dict, ['training_split', 'training_dataset','use_tags', 'n_samples'])	

		tags_str = 'with_tags' if spec_dict['use_tags'] else 'no_tags'
		n_str = 'n='+str(spec_dict['n_samples'])

		path = join(config.exp_dir, spec_dict['task_phase'], n_str, spec_dict['training_split'] + '_' + spec_dict['training_dataset'] + '_'  + tags_str) 

	
	elif spec_dict['task_phase'] == 'train':

		confirm_values_are_none(spec_dict,['context_width', 'test_dataset','test_split'])
		confirm_values_are_not_none(spec_dict,['task_phase', 'model_type', 'training_split','training_dataset', 'use_tags'])

		tags_str = 'with_tags' if spec_dict['use_tags'] else 'no_tags'
		n_str = 'n='+str(spec_dict['n_samples'])

		path = join(config.exp_dir, spec_dict['task_phase'], n_str, spec_dict['training_split'] + '_' + spec_dict['training_dataset'] + '_'  + tags_str + 'x' + spec_dict['model_type']) 

	elif spec_dict['task_phase'] in ('fit','eval'):


		confirm_values_are_not_none(spec_dict, ['training_split', 'training_dataset','use_tags', 'n_samples', 'test_split', 'test_dataset', 'context_width'])
		validate_test_params(spec_dict)
				
		tags_str = 'with_tags' if spec_dict['use_tags'] else 'no_tags'
		n_str = 'n='+str(spec_dict['n_samples'])

		path = join(config.exp_dir, spec_dict['task_phase'], n_str, spec_dict['training_split'] + '_' + spec_dict['training_dataset'] + '_'  + tags_str + 'x' + spec_dict['model_type'] + '_' +  spec_dict['test_split'] + '_' + spec_dict['test_dataset'] +'_' + str(spec_dict['context_width']))

	else:
		raise ValueError('Task phase not recognized. Must be one of '+config.task_phases)

	return(path)


def get_file_identifier(spec_dict):	
	tags_str = 'with_tags' if spec_dict['use_tags'] else 'no_tags'    
	if spec_dict['task_phase'] == 'train':        	
		path =  spec_dict['task_phase']+'_'+spec_dict['training_split']+'_'+spec_dict['training_dataset']+'_'+tags_str+'_'+spec_dict['model_type']
	elif spec_dict['task_phase'] == 'fit':        	
		path =  spec_dict['task_phase']+'_'+spec_dict['training_split']+'_'+spec_dict['training_dataset']+'_'+tags_str+'_'+spec_dict['model_type']+'_'+spec_dict['test_split']+'_'+spec_dict['test_dataset']+'_'+str(spec_dict['context_width'])
	elif spec_dict['task_phase'] == 'eval':        	
		path =  spec_dict['task_phase']+'_'+spec_dict['training_split']+'_'+spec_dict['training_dataset']+'_'+tags_str+'_'+spec_dict['model_type']+'_'+spec_dict['test_split']+'_'+spec_dict['test_dataset']+'_'+str(spec_dict['context_width'])
	else:
		raise NotImplementedError
	return(path)


def get_slurm_script_name(spec_dict):
		# formerly get_slurm_script_path
		#<training_split>_<training_dataset>(x<tags>)(x<model_type>)(x<test_split>_<test_dataset>_<context_width>)

		path = get_file_identifier(spec_dict)+'.sh'	
		return(path)


def get_sample_csv_path(task_phase_to_sample_for, split, dataset, data_type, age = None, n=None):    

    assert ( (age is None) and (task_phase_to_sample_for == 'fit' or split == 'Providence') ) or ( (age is not None) and (task_phase_to_sample_for == 'eval') )
    age_str = f'_{float(age)}' if age is not None else ''
    
    assert data_type in ['success', 'yyy'], "Invalid data type requested of sample path: choose one of {success, yyy}."
    
    # where should the sampling csvs get stored? at samples for a run,
    # dvided by split and then by datase
    sample_folder =  get_directory({
		"test_split" : None, 
		"training_split" : split,
		"test_dataset" : None,
		"training_dataset" : dataset,
		"model_type" : None,
		"n_samples" : n,
		"use_tags" : None,
		"context_width" : None,        
		"task_name" : None,
		"task_phase" : 'sample'})

    if not exists(sample_folder):
    	os.makedirs(sample_folder)
    
    this_data_path = join(sample_folder, f'{task_phase_to_sample_for}_{data_type}_utts_{str(n)}{age_str}.csv')
    
    return this_data_path