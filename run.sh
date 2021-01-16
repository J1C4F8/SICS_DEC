#!/usr/bin/env bash -l

# setup conda environment
conda env create -f sics.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sics-gpu

# run the script
python3 clustering.py