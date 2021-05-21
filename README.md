# child-directed-listening
Analyses for "Child Directed Listening" at ICIS 2020 and "Child-directed Listening: How Caregiver Inference Enables Children's Early Verbal Communication" submitted to CogSci 2021.

Users are strongly encouraged to install the Python packages in a virtual enviroment and use the virtual environment as a Jupyter kernel.

### CogSci 2021: Child-directed Listening: How Caregiver Inference Enables Children's Early Verbal Communication

Primary analyses are in `BERT Prediction xxx.ipynb`, supported by the functions in `transfomers_bert_completions.py`


# Retrieving completions from BERT

Helper functions in `transfomers_bert_completions.py` are structured as follows:
![function relationships in transformers retrieval code](figures/transformers_retrieval.png)

Blue and red indicate top-level functions: 

- `sample_models_across_time` (red) selects a number of communicative successes and failures from each time period, retrieves prior probabilities from BERT models and unigram models as matrices, computes an edit-distance based likelihood (also a matrix), and does element-wise multiplication and row normalization to get the posteriors. 

- `sample_across_models` (blue) is similar to the above, except it accepts a number of communicative successes and failures, and it can iterate over a range of beta values to compute the likelihood values (where higher beta assigns lower probabilities to words at a higher edit distance) 
