# child-directed-listening

Analyses for "Child Directed Listening".

Users are strongly encouraged to install the Python packages in a virtual enviroment and use the virtual environment as a Jupyter kernel.

Primary analyses are in `Models across time analyses.ipynb`, supported by the functions in `utils/transfomers_bert_completions.py`, which are executed by the files in the repository root that are named `run_*.py`.

# Generating results

Note that this assumes use of a SLURM system.

Start on local machine.
1. Run `tier_1_data_gen.sh`.
2. Rsync everything according to the directions in that .sh file.
On your SLURM machine:
3. `Run tier_2a_non_child_train_shelf_scores.sh`
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

