#!/bin/bash

#$ -l tmem=12G
#$ -l h_vmem=12G
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
echo 'which nvcc': which nvcc
echo 'nvcc --version:'; nvcc --version
echo 'which python: '; which python
echo 'python --version:'; python --version

conda activate shahin_msc24

## Only need to run this command once to install. In fact, I ran this via qrsh (after running conda activate shahin_msc).
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda list | grep -i "cuda\|cudnn\|torch"
#!/usr/bin/env bash
#source ~/.bashrc
#source ~/miniconda3/bin/activate shahin_msc24

#export PYTHONPATH=$PYTHONPATH:~/diffSock_HPC
#export PYTHONPATH=$PYTHONPATH:~/miniconda3/envs/shahin_msc24/lib/python3.12/site-packages
#export LD_LIBRARY_PATH=~/miniconda3/envs/shahin_msc24/lib:$LD_LIBRARY_PATH

## Set the CUDA_LAUNCH_BLOCKING environment variable for debugging
#export CUDA_LAUNCH_BLOCKING=1

cd ~/diffSock_HPC/rnatrain_DJ_code || exit
python pytorch_rnafold_allatomclustrain_singlegpu.py