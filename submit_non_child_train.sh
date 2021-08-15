#!/bin/bash
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/models/age/old
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/models/age/young
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/models/all/all
FILES="./scripts_non_child_train/*/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done
