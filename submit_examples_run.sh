#!/bin/bash
mkdir -p experiments/no_versioning/scores/n=500/val/all/all/
FILES="./scripts_examples_run/*"
for f in $FILES
do
	echo "Processing $f file..."
	sbatch $f
	cat "$f"
done