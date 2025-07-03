#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N bert_baseline_lstm
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output.log
#$ -e log/error.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

grep -q '^DENOISE_LABELS=' .env && \
  sed -i'' 's/^DENOISE_LABELS=.*/DENOISE_LABELS=False/' .env || \
  echo 'DENOISE_LABELS=False' >> .env
grep -q '^HEAD_TYPE=' .env && \
  sed -i'' 's/^HEAD_TYPE=.*/HEAD_TYPE=lstm/' .env || \
  echo 'HEAD_TYPE=lstm' >> .env

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

# Train
python src/pipeline/bert_train_pipeline.py

# Test
python src/pipeline/bert_test_pipeline.py