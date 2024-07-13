"""
TRAINING LAUNCHER SCRIPT
------------------------

INPUT = amino acid sequence (1-letter FASTA format).

OUTPUT = XYZ coordinates for atoms that are seen, otherwise `<NA>`.
"""
import tokeniser as tk
from data_layer import data_handler as dh


if __name__ == '__main__':

    pdb_ids = dh.get_list_of_pdbids_of_local_single_domain_cifs()
    cif_count = len(pdb_ids)
    dh.write_list_to_space_separated_txt_file(list_to_write=pdb_ids,
                                              file_name=f'pdb_ids_list/SD_pdbids_{cif_count}.txt')

    # tk.write_tokenised_cif_to_csv(pdb_ids)

