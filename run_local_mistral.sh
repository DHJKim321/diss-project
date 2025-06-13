#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N eval_local_mistral
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate diss

# Run the program
python src/pipeline/local_prompt_pipeline.py