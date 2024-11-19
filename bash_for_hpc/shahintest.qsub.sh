#!/bin/bash

#$ -l tmem=100M
#$ -l h_vmem=100M
#$ -l h_rt=360

#$ -S /bin/bash
#$ -j y
#$ -N SHAHINTESTJOB
#$ -cwd

hostname
date
pwd

#!/usr/bin/env bash
source ~/.bashrc
source ~/miniconda3/bin/activate shahin_msc24

## Limit MKL threads to prevent memory issues - suggested as a possible solution to memory errors by ChatGPT
#export MKL_NUM_THREADS=1
#export OMP_NUM_THREADS=1

export PYTHONPATH=$PYTHONPATH:~/diffSock_HPC
export PYTHONPATH=$PYTHONPATH:~/miniconda3/envs/shahin_msc24/lib/python3.12/site-packages

#!/usr/bin/env python3
#python3 ~/diffSock_HPC/helloworld_for_HPC.py
python3 ~/diffSock_HPC/rnatrain_DJ_code/hello_from_rnatrain.py