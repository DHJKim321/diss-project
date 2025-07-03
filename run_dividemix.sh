#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N dividemix
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -l gpu=1
#$ -o log/output.log
#$ -e log/error.log

# Initialise the environment modules
. /etc/profile.d/modules.sh

grep -q '^HEAD_TYPE=' .env && \
  sed -i'' 's/^HEAD_TYPE=.*/HEAD_TYPE=linear/' .env || \
  echo 'HEAD_TYPE=linear' >> .env

grep -q '^USE_IMDB=' .env && \
  sed -i'' 's/^USE_IMDB=.*/USE_IMDB=False/' .env || \
  echo 'USE_IMDB=False' >> .env

# Load Python
module load anaconda

conda activate /exports/eddie/scratch/s2017594/conda-envs/diss

python src/pipeline/dividemix_pipeline.py