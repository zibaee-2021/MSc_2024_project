"""
TRAINING LAUNCHER SCRIPT
------------------------

INPUT = amino acid sequence (1-letter FASTA format).

OUTPUT = XYZ coordinates for atoms that are seen, otherwise `<NA>`.
"""
import tokeniser as tk
from data_layer import data_handler as dh


if __name__ == '__main__':

    pdb_ids = dh.get_list_of_locally_downloaded_pdb_ids()
    pdbids_string = ' '.join(pdb_ids)

    with open('../data/pdb_ids_list/pdbs_ids.txt', 'w') as f:
        f.write(pdbids_string)

    # tk.write_tokenised_cif_to_csv(pdb_ids)

