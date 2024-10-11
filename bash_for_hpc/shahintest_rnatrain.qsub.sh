#!/bin/bash

#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=3600
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N SHAHINTESTJOB_RNATRAIN
#$ -cwd
#$ -V

hostname
date
pwd

echo 'nvidia-smi:'; nvidia-smi
echo 'nvcc --version:'; nvcc --version
echo 'which python: '; which python

conda activate shahin_msc24
#!/usr/bin/env bash
#source ~/.bashrc
#source ~/miniconda3/bin/activate shahin_msc24

#export PYTHONPATH=$PYTHONPATH:~/diffSock
#export PYTHONPATH=$PYTHONPATH:~/miniconda3/envs/diffSock/lib/python3.12/site-packages
#export LD_LIBRARY_PATH=~/miniconda3/envs/shahin_msc24/lib:$LD_LIBRARY_PATH  #  tells where to first look for libraries

cd ~/diffSock/rnatrain
python pytorch_rnafold_allatomclustrain_singlegpu.py