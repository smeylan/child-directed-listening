#!/bin/bash
mkdir -p /om2/user/wongn/child-directed-listening/experiments/lr_search/models/age/old
mkdir -p /om2/user/wongn/child-directed-listening/experiments/lr_search/models/age/young
mkdir -p /om2/user/wongn/child-directed-listening/experiments/lr_search/models/all/all
FILES="./scripts_non_child_train_search/*/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done
