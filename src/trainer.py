"""
TRAINING LAUNCHER SCRIPT
------------------------

INPUT = amino acid sequence (1-letter FASTA format).

OUTPUT = XYZ coordinates for atoms that are seen, otherwise `<NA>`.
"""
from enum import Enum
from src.preprocessing_funcs import tokeniser as tk
from data_layer import data_handler as dh


class Paths(Enum):
    pdbid_dir = 'PDBid'


if __name__ == '__main__':

    SD_PDBids = dh.get_list_of_pdbids_of_local_single_domain_cifs()
    cif_count = len(SD_PDBids)
    dh.write_list_to_space_separated_txt_file(list_to_write=SD_PDBids,
                                              file_name=f'{Paths.pdbid_dir.value}/SD_{cif_count}.txt')

    # tk.write_tokenised_cif_to_csv(pdb_ids)

