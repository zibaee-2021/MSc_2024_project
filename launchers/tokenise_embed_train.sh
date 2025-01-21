#!/usr/bin/env bash
source ~/.bashrc
source ~/miniconda3/bin/activate diffSock

export PYTHONPATH=$PYTHONPATH:$HOME/diffSock
export PYTHONPATH=$PYTHONPATH:$HOME/diffSock/data_layer
export PYTHONPATH=$PYTHONPATH:$HOME/diffSock/src
export PYTHONPATH=$PYTHONPATH:$HOME/diffSock/src/diffusion

#!/usr/bin/env python3
python3 ~/PycharmProjects/MSc_project/diffSock/src/diffusion/tokenise_embed_train.py pdbchains_565
