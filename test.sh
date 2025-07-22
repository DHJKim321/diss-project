#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N bert_loss_test
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output_bert_loss_test.log
#$ -e log/error_bert_loss_test.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

# Run the program
python src/pipeline/bert_loss_pipeline.py