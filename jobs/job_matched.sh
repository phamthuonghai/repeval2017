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
#$ -o ./jobs/log_matched.out
#
# Run job through bash shell
#$ -S /bin/bash
#

PYTHON=$(pwd)/venv/bin/python
source $(pwd)/venv/bin/activate

${PYTHON} -u ./nli.py

mv ./jobs/log_matched.out ./jobs/log_matched_$(date +%Y-%m-%d-%H-%M-%S).out
