# rsync results from SLURM back to the local machine
# to be run on the local (non-SLURM) machine

rsync -az --progress ${SLURM_USERNAME}:${CDL_SLURM_ROOT} ./experiments