#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N mistral
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output_mistral.log
#$ -e log/error_mistral.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

# Run the program
python src/pipeline/local_prompt_pipeline.py