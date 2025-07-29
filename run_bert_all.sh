#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N bert_baseline_ag_0.8
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=100G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output_bert_ag_0.8.log
#$ -e log/error_bert_ag_0.8.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

python src/pipeline/bert_pipeline.py