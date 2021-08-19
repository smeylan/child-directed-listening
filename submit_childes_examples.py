#!/bin/bash
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/age/old
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/age/young
mkdir -p /om2/user/wongn/child-directed-listening/experiments/no_versioning/scores/n=500/val/all/all
FILES="./scripts_examples_run_childes/finetune/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done
