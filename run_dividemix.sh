#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N dividemix_50_aeda
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output_dividemix_aeda.log
#$ -e log/error_dividemix_aeda.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

python src/pipeline/dividemix_pipeline.py