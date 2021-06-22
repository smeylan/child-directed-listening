python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file /home/stephan/notebooks/child-directed-listening/data/train.txt \
    --validation_file /home/stephan/notebooks/child-directed-listening/data/validation.txt \
    --vocab_file /home/stephan/notebooks/child-directed-listening/data/vocab.csv \
    --do_train \
    --do_eval \
    --output_dir /home/stephan/python/bert-finetuned-on-childes/model_output/ \
    --overwrite_output_dir
