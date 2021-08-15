#!/bin/bash
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Alex
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Ethan
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Lily
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Naima
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/Violet
mkdir -p /om2/user/wongn/child-directed-listening/experiments/goal_to_convergence/scores/n=500/val/child/William
FILES="./scripts_child_train/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done
