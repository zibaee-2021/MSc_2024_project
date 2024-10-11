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

export PYTHONPATH=$PYTHONPATH:~/diffSock
export PYTHONPATH=$PYTHONPATH:~/miniconda3/envs/diffSock/lib/python3.12/site-packages

#!/usr/bin/env python3
#python3 ~/diffSock/helloworld_for_HPC.py
python3 ~/diffSock/rnatrain/hello_from_rnatrain.py