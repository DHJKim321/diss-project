#!/bin/sh
# Grid Engine options
#$ -N gmm_bert
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output.log
#$ -e log/error.log

ENV_FILE=".env"

KEY1="DENOISE_LABELS"
VALUE1="True"
if grep -q "^$KEY1=" "$ENV_FILE"; then
    sed -i "s|^$KEY1=.*|$KEY1=$VALUE1|" "$ENV_FILE"
else
    echo "$KEY1=$VALUE1" >> "$ENV_FILE"
fi

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

# Train
python src/pipeline/bert_train_pipeline.py

# Test
python src/pipeline/bert_test_pipeline.py