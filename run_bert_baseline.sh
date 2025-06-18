#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N eval_local_mistral
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output.log
#$ -e log/error.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

# Train
python src/pipeline/bert_train_pipeline.py

# Test
python src/pipeline/bert_test_pipeline.py