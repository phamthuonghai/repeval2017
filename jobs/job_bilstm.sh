#!/usr/bin/env bash
#
# Select Q
#$ -q main.q
#
# Your job name
#$ -N repeval_matched
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o ./jobs/log_bilstm.out
#
# Run job through bash shell
#$ -S /bin/bash
#

PYTHON=$(pwd)/venv/bin/python
source $(pwd)/venv/bin/activate
DATE=$(date +%Y-%m-%d-%H-%M-%S)

${PYTHON} -u ./nli.py --save_path results/bilstm_${DATE} --train_embed

mv ./jobs/log_bilstm.out ./jobs/log_bilstm_${DATE}.out
