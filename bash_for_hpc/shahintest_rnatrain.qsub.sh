#!/bin/bash

#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=3600
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N SHAHINTESTJOB_RNATRAIN
#$ -cwd

hostname
date
pwd

which python

#!/usr/bin/env bash
source ~/.bashrc
source ~/miniconda3/bin/activate shahin_msc24

export PYTHONPATH=$PYTHONPATH:~/diffSock
export PYTHONPATH=$PYTHONPATH:~/miniconda3/envs/diffSock/lib/python3.12/site-packages
export LD_LIBRARY_PATH=~/miniconda3/envs/shahin_msc24/lib:$LD_LIBRARY_PATH

#!/usr/bin/env python3
python3 ~/diffSock/rnatrain/pytorch_rnafold_allatomclustrain_singlegpu.py