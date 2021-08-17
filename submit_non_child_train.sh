#!/bin/bash
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/models/age/old
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/models/age/young
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_search_retrain_default/models/all/all
FILES="./scripts_non_child_train/*/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done
