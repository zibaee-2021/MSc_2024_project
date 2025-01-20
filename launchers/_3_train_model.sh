#!/usr/bin/env bash
source ~/.bashrc
source ~/miniconda3/bin/activate diffSock

export PYTHONPATH=$PYTHONPATH:$HOME/diffSock

#!/usr/bin/env python3
python3 ~/PycharmProjects/MSc_project/diffSock/src/diffusion/pytorch_protfold_allatomclustrain_singlegpu.py
