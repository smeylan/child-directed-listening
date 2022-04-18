from src.utils import configuration, split_gen, scripts
from src.gen import gen_training_scripts, gen_eval_scripts
config = configuration.Config()

def gen_fitting_commands(
        split_name,
        training_split_name, 
        dataset_name,
        training_dataset_name,
        model_type,
        use_tags,
        context_width,
        task_name):
    
    your_model_path = split_gen.get_split_folder(split_name, training_dataset_name, config.model_dir)
    
    # ---------- begin new code
    
    # Generate the appropriate header and the slurm folder
    
    slurm_folder = scripts.get_slurm_folder(split_name, training_dataset_name, task_name)
    
    mem_alloc_gb, time_alloc_hrs,  n_tasks, cpus_per_task = gen_training_scripts.get_training_alloc(split_name)
    
    header_commands = scripts.gen_command_header(mem_alloc_gb = mem_alloc_gb, time_alloc_hrs = time_alloc_hrs,
        n_tasks = n_tasks,
        cpus_per_task = cpus_per_task,
        slurm_folder = slurm_folder,
        slurm_name = f'training_beta_tags={use_tags}', 
        two_gpus = False)
    commands = header_commands


    this_model_dir = '/'.join(gen_training_scripts.models_get_split_folder(split_name, training_dataset_name, use_tags).split('/')[:-1])
    

    sing_header = scripts.gen_singularity_header()   
    
    run_commands = [f"{sing_header} {gen_eval_scripts.get_one_python_command('src/run/run_beta_search.py', split_name, dataset_name , use_tags, context_width, model_type, training_dataset_name, training_split_name)[1]}\n"]    
        
    # Put the copy commands between the header and the actual python runs.
    commands += run_commands
    
    filename = scripts.get_script_name(split_name, dataset_name, use_tags, context_width, training_dataset_name, model_type)


    return filename, commands