#!/usr/bin/env bash
source ~/.bashrc
source ~/miniconda3/bin/activate diffSock

export PYTHONPATH=$PYTHONPATH:$HOME/diffSock
export PYTHONPATH=$PYTHONPATH:$HOME/diffSock/data_layer

#!/usr/bin/env python3
python3 ~/PycharmProjects/MSc_project/diffSock/src/pLM/plm_embedder.py
