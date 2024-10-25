"""
INFERENCE LAUNCHER SCRIPT
-------------------------

INPUT = amino acid sequence (1-letter FASTA format).

OUTPUT = all possible atoms (obviously with no coordinates).
"""
from src.preprocessing_funcs import faa_to_atoms

if __name__ == '__main__':

    id_to_atomic_sequence = faa_to_atoms.translate_aa_to_atoms(uniprot_ids=[])

