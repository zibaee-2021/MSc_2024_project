"""
TRAINING LAUNCHER SCRIPT
------------------------

INPUT = amino acid sequence (1-letter FASTA format).

OUTPUT = XYZ coordinates for atoms that are seen, otherwise `<NA>`.
"""
# from enum import Enum
from src.preprocessing_funcs import tokeniser as tk
from data_layer import data_handler as dh


# class Path(Enum):
#     pdbid_dir = 'diffusion/diff_data/PDBid_list'

# TODO
# ONLY NEED TO RUN THESE ONCE FOR A PARTICULAR DATASET
# if __name__ == '__main__':
#     pass
