# child-directed-listening

Analyses for "Child Directed Listening".

Users are strongly encouraged to install the Python packages in a virtual enviroment and use the virtual environment as a Jupyter kernel.

Primary analyses are in `Models across time analyses.ipynb`, supported by the functions in `utils/transfomers_bert_completions.py`, which are executed by the files in the repository root that are named `run_*.py`.

# Generating results

Note that this assumes use of a SLURM system for GPU access. Both the local and the remote machine should have Python 3.6. The general organization is shell scripts which call jupyter notebook `nbconvert` to output notebooks where all cells have been run. The notebooks can be inspected, and the scripts generate figures and tables used in the paper.

0. Set the following environment variables on both your local machine and on the SLURM machine
```
export SLURM_USERNAME="*****@******"  
export CDL_SLURM_USER="smeylan"  
export CDL_SLURM_ROOT="~/om2/projects/nicole/child_repo_split/"  
export CDL_CONFIG_PATH="stephan_configuration.json"  
export CDL_SINGULARITY_PATH="/om2/user/wongn/vagrant/trans-pytorch-gpu"  
```
Above values are examples. Set them as follows:

`SLURM_USERNAME` is username and domain of the SLURM login node.  
`CDL_SLURM_USER` is the username of the SLURM user  
`CDL_SLURM_ROOT` is the path relative to your user folder on the SLURM machine where all results will reside  
`CDL_CONFIG_PATH` is the path to the json configuration file (similar to command line arguments)  
`CDL_SINGULARITY_PATH` is the path to the Singularity image on the SLURM machine (with Transformers, pytorch, etc.)  
 
2. Set up a virtual environment on the local machine and activate it `virtualenv -p python3.6 cdl_env && source cdl_env/bin/activate`
3. Install the Python dependencies on the local machine `pip3 install -r requirements.txt`
4. Register the virtual environment as a kernel with Jupyter: `python -m ipykernel install --user --name=child-listening-env`
5. From `~/.jupyter/jupyter_nbconvert_config.json` remove the entry `"postprocessor_class": "jupyter_contrib_nbextensions.nbconvert_support.EmbedPostProcessor"`
6. On your local machine, run `./tier_1_data_gen.sh`. The end of this script starts an rsync job that copies the generated files to the SLURM machine; this may require your password
8. On your SLURM machine, run `sbatch tier_2a_non_child_train_shelf_scores.sh`. This FIXME -- WHAT DOES THIS DO?
After the computations have fully completed (not after the .sh completes submitting the jobs, but after you have confirmed that the program executed completely), run `tier_2b_finetune_scores_child_train.sh`.
4. After the computations have fully completed, run `tier_2c_child_cross.sh`.
4. Follow the rsync directions in 2c file to rsync the `experiments` folder back to your local machine.
5. When the relevant notebooks are complete, run `tier_3_analysis.sh`.
6. Locate your analyses in the following notebooks:
    a. `Examples for Success and Failure Table.ipynb`
    b. `Models across time analyses.ipynb`
    c. `Child evaluations.ipynb`
    d. `Prevalence analyses.ipynb`

Note that all commands do not necessarily have a flexible OpenMind (SLURM) user specified.


# Retrieving completions from BERT

Most functions that support scoring computations can be found in `utils/transformers_bert_completions.py`.

Note that running these functions via the scripts requires rsyncs as described in the .sh files.

Below is an overview of how the token scores are generated:
![function relationships in transformers retrieval code](figures_info/codebase_diagram.jpg)

The top-level functions and function calls are in the following files:

- `run_models_across_time.py` computes scores of tokens from samples that have been drawn across time. It will eventually call `success_and_failures_across_time_per_model`. This function retrieves prior probabilities from BERT models and unigram models as matrices, computes an edit-distance based likelihood (also a matrix), and does element-wise multiplication and row normalization to get the posteriors.

- `run_beta_search.py` will call ``optimize_beta``, which calls ``sample_across_models``. The latter is similar to ``successes_and_failures_across_time_per_model``, above. However, it computes the optimal beta within a search space of beta values, given a sample of communicative successes, which are not drawn from across time. It computes the likelihood values (where higher beta assigns lower probabilities to words at a higher edit distance).

- `run_child_cross.py` will call ``score_cross_prior``, which will calculate the scores per token similar to the across time and beta functions described above. However, it will do so by loading a model finetuned on a certain child, and data associated with another child (possibly, but not necessarily, different children).

# Example commands

For training (non-child):

`python3 run_mlm.py --train_file /net/vast-storage/scratch/vast/cpl/wongn/child_split/child-directed-listening/finetune/age/old/train_no_tags.txt --validation_file /net/vast-storage/scratch/vast/cpl/wongn/child_split/child-directed-listening/finetune/age/old/val_no_tags.txt --cache_dir ~/.cache/$SLURM_JOB_ID --output_dir /net/vast-storage/scratch/vast/cpl/wongn/child_split/child-directed-listening/experiments/no_versioning/models/age/old/no_tags --do_eval  --do_train  --eval_steps 1 --evaluation_strategy steps --learning_rate 5e-05 --load_best_model_at_end  --logging_steps 1 --logging_strategy steps --metric_for_best_model eval_loss --model_name_or_path bert-base-uncased --num_train_epochs 3 --overwrite_output_dir  --per_device_eval_batch_size 8 --per_device_train_batch_size 8 --save_steps 1 --save_strategy steps --save_total_limit 1 --max_train_samples 10 --max_eval_samples 10`

For BERT model scoring:
`python3 run_models_across_time.py --split age --dataset old --context_width 0 --use_tags False --model_type childes`

For unigram scoring:
`python3 run_beta_search.py --split all --dataset all --context_width 0 --use_tags False --model_type flat_unigram`
`python3 run_models_across_time.py --split all --dataset all --context_width 0 --use_tags False --model_type flat_unigram`

For child scoring:
`python3 run_child_cross.py --data_child Alex --prior_child Alex`

