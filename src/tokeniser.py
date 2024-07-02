import cif_parser as parser
import json
import numpy as np
from enum import Enum

if __name__ == '__main__':

    class Tokens(Enum):
        AA_INDEX = 'aa_index'  # `S_seq_id` (the position of the amino acid)
        ATOM_INDEX = 'atom_index'  # `A_id` (the position of the atom)

        AA_LABEL_NUM = 'aa_label_num'  # `A_label_comp_id` enumerated (the amino acid)
        ATOM_LABEL_NUM = 'atom_label_num'  # `A_label_atom_id` enumerated (the atom)

        BB_INDEX = 'bb_index'  # NOT CLEAR WHAT THIS IS .. BACKBONE ATOMS ?
        MEAN_COORDS = 'mean_xyz'
        MEAN_CORR_X = 'mean_corrected_x'
        MEAN_CORR_Y = 'mean_corrected_y'
        MEAN_CORR_Z = 'mean_corrected_z'


    pdb_id = '4hb1'
    pdf_cif = parser.parse_cif(local_cif_file=f'../data/cifs/{pdb_id}.cif')
    with open('../data/jsons/unique_atoms_only_enumerated.json', 'r') as json_f:
        atoms_enumerated = json.load(json_f)
    with open('../data/jsons/aas_enumerated.json', 'r') as json_f:
        aas_enumerated = json.load(json_f)

    # Amino acid indices
    aa_index = np.asarray(pdf_cif[parser.CIF.S_seq_id.value].tolist(), dtype=np.uint16)

    # Atom indices
    pdf_cif[parser.CIF.A_id.value].fillna(0, inplace=True)
    atom_index = np.asarray(pdf_cif[parser.CIF.A_id.value].tolist(), dtype=np.uint16)

    # Amino acid labels enumerated
    pdf_cif[Tokens.AA_LABEL_NUM.value] = pdf_cif[parser.CIF.A_label_comp_id.value].map(aas_enumerated).astype('Int64')
    pdf_cif[Tokens.AA_LABEL_NUM.value].fillna(255, inplace=True)
    aa_label_num = np.asarray(pdf_cif[Tokens.AA_LABEL_NUM.value].tolist(), dtype=np.uint8)

    # Atom labels enumerated
    pdf_cif[Tokens.ATOM_LABEL_NUM.value] = pdf_cif[parser.CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')
    pdf_cif[Tokens.ATOM_LABEL_NUM.value].fillna(255, inplace=True)
    atom_label_num = np.asarray(pdf_cif[Tokens.ATOM_LABEL_NUM.value].tolist(), dtype=np.uint8)

    # Atomic xyz coordinates
    pdf_cif[Tokens.MEAN_COORDS.value] = pdf_cif[[parser.CIF.A_Cartn_x.value,
                                                 parser.CIF.A_Cartn_y.value,
                                                 parser.CIF.A_Cartn_z.value]].mean(axis=1)
    pdf_cif[Tokens.MEAN_CORR_X.value] = pdf_cif[parser.CIF.A_Cartn_x.value] - pdf_cif[Tokens.MEAN_COORDS.value]
    pdf_cif[Tokens.MEAN_CORR_Y.value] = pdf_cif[parser.CIF.A_Cartn_y.value] - pdf_cif[Tokens.MEAN_COORDS.value]
    pdf_cif[Tokens.MEAN_CORR_Z.value] = pdf_cif[parser.CIF.A_Cartn_z.value] - pdf_cif[Tokens.MEAN_COORDS.value]

    mean_corrected_xyz = pdf_cif[[Tokens.MEAN_CORR_X.value, Tokens.MEAN_CORR_Y.value, Tokens.MEAN_CORR_Z.value]]
    mean_corr_xyz_npy = mean_corrected_xyz.to_numpy()
    pass

    tokens = (Tokens.AA_LABEL_NUM.value, Tokens.ATOM_LABEL_NUM.value, Tokens.AA_INDEX.value, pdb_id, mean_corr_xyz_npy)
