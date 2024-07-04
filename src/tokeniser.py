import cif_parser as parser
from cif_parser import CIF
import json
import numpy as np
import pandas as pd
from enum import Enum


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'  # `A_label_comp_id` enumerated (the amino acid)
    ATOM_LABEL_NUM = 'atom_label_num'  # `A_label_atom_id` enumerated (the atom)
    BB_INDEX = 'bb_index'  # NOT CLEAR WHAT THIS IS .. BACKBONE ATOMS ?  ?  ?
    MEAN_COORDS = 'mean_xyz'  # mean of x y z coordinates for each atom
    MEAN_CORR_X = 'mean_corrected_x'  # x coordinates for each atom subtracted by the mean of xyz coordinates
    MEAN_CORR_Y = 'mean_corrected_y'  # (as above) but for y coordinates
    MEAN_CORR_Z = 'mean_corrected_z'  # (as above) but for z coordinates


def _read_enumeration_mappings():
    with open('../data/jsons/unique_atoms_only_enumerated.json', 'r') as json_f:
        atoms_enumerated = json.load(json_f)
    with open('../data/jsons/aas_enumerated.json', 'r') as json_f:
        aas_enumerated = json.load(json_f)
    return atoms_enumerated, aas_enumerated


def _write_to_csv(pdb_id: str, pdf: pd.DataFrame):
    pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.csv', index=False, na_rep='null')
    pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.ssv', sep=' ', index=False, na_rep='null')  # space-separated
    pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.tsv', sep='\t', index=False, na_rep='null')  # tab-separated
    pdf_easy_read = pdf.rename(columns={CIF.S_seq_id.value: 'SEQ_ID',
                                        CIF.S_mon_id.value: 'RESIDUES',
                                        CIF.A_id.value: 'ATOM_ID',
                                        CIF.A_label_atom_id.value: 'ATOMS',
                                        CIF.A_Cartn_x.value: 'X',
                                        CIF.A_Cartn_y.value: 'Y',
                                        CIF.A_Cartn_z.value: 'Z'})
    pdf_easy_read.to_csv(path_or_buf=f'../data/tokenised/easyRead_{pdb_id}.tsv', sep='\t', index=False, na_rep='null')


def write_tokenised_cif_to_csv(pdb_ids=None) -> None:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `../data/tokenised/`.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, local_cif_file=f'../data/cifs/{pdb_id}.cif')
        atoms_enumerated, aas_enumerated = _read_enumeration_mappings()

        # # Amino acid indices
        # aa_index = np.asarray(pdf_cif[CIF.S_seq_id.value].tolist(), dtype=np.uint16)
        # # Atom indices
        # pdf_cif[CIF.A_id.value].fillna(0, inplace=True)
        # atom_index = np.asarray(pdf_cif[CIF.A_id.value].tolist(), dtype=np.uint16)

        # Amino acid labels enumerated
        pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')
        pdf_cif[ColNames.AA_LABEL_NUM.value].fillna(255, inplace=True)
        pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[ColNames.AA_LABEL_NUM.value].astype('uint8')

        # Atom labels enumerated
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')
        pdf_cif[ColNames.ATOM_LABEL_NUM.value].fillna(255, inplace=True)
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[ColNames.ATOM_LABEL_NUM.value].astype('uint8')

        # Atomic xyz coordinates
        pdf_cif[ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                       CIF.A_Cartn_y.value,
                                                       CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif[ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]

        # alpha_carbon_indices = np.where(pdf_cif[CIF.A_label_atom_id.value] == 'CA',
        #                                 pdf_cif[CIF.A_id.value], np.nan)
        # alpha_carbon_indices = alpha_carbon_indices[~np.isnan(alpha_carbon_indices)]
        # alpha_carbon_indices = np.asarray(alpha_carbon_indices, dtype=np.uint16)
        _write_to_csv(pdb_id, pdf_cif)


