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
#$ -o ./jobs/log_job_bilstm.log
#
# Run job through bash shell
#$ -S /bin/bash
#

PYTHON=/users/ud2017/anaconda2/bin/python
DATE=$(date +%Y-%m-%d-%H-%M-%S)

${PYTHON} -u ./nli.py --lstm-pooling max --save-path results/bilstm_maxpooling > ./jobs/log_bilstm_maxpooling.out 2>&1 &
${PYTHON} -u ./nli.py --shared-encoder --save-path results/bilstm_sharedencoder > ./jobs/log_bilstm_sharedencoder.out 2>&1 &
wait
${PYTHON} -u ./nli.py --save-path results/bilstm > ./jobs/log_bilstm.out 2>&1 &
${PYTHON} -u ./nli.py --train-embed --save-path results/bilstm_trainembed > ./jobs/log_bilstm_trainembed.out 2>&1 &
wait
${PYTHON} -u ./nli.py --lstm-pooling avg --save-path results/bilstm_avgpooling > ./jobs/log_bilstm_avgpooling.out 2>&1 &
wait
