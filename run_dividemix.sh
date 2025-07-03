#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N eval_local_mistral
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output.log
#$ -e log/error.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

ENV_FILE=".env"

KEY1="HEAD_TYPE"
VALUE1="linear"
if grep -q "^$KEY1=" "$ENV_FILE"; then
    sed -i "s|^$KEY1=.*|$KEY1=$VALUE1|" "$ENV_FILE"
else
    echo "$KEY1=$VALUE1" >> "$ENV_FILE"
fi

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

python src/pipeline/dividemix_pipeline.py