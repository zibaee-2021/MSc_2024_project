"""
CALLS THE CIF PARSER TO READ IN AND PARSE THE CIF FILE TO EXTRACT THE FOLLOWING 14 FIELDS

These 14 fields are used and end up in a 14-column dataframe. A description of what they are all used for is given here
and below (I am happy to repeat myself in an effort to reduce the chance of mistakes due to confusing names).

atom_site:
    group_PDB,          # 'ATOM' or 'HETATM'    - Filter on this then remove.
    auth_seq_id,        # residue position      - used to join with S_pdb_seq_num, then remove.
    label_comp_id,      # residue (3-letter)    - used to sanity-check with S_mon_id, then remove.
    id,                 # atom position         - sort on this, keep.
    label_atom_id,      # atom                  - keep
    label_asym_id,      # chain                 - join on this, sort on this, keep.
    Cartn_x,            # atom x-coordinates
    Cartn_y,            # atom y-coordinates
    Cartn_z,            # atom z-coordinates
    occupancy           # occupancy

_pdbx_poly_seq_scheme:
    seq_id,             # residue position      - sort on this, keep.
    mon_id,             # residue (3-letter)    - used to sanity-check with A_label_comp_id, keep.
    pdb_seq_num,        # residue position      - join to A_auth_seq_id, then remove.
    asym_id,            # chain                 - join on this, sort on this, then remove.
"""


import os

import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from src.preprocessing_funcs.cif_parser import CIF
from data_layer import data_handler as dh
from enum import Enum

# TODO Traceback (most recent call last):
#   File "/home/shahin/PycharmProjects/MSc_project/diffSock/src/general_utility_methods/tokeniser.py", line 78, in <module>
#   File "/home/shahin/PycharmProjects/MSc_project/diffSock/src/general_utility_methods/tokeniser.py", line 71, in parse_tokenise_cif_write_to_flatfile_to_pdf


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'       # Enumerated residues, mapped from `A_label_comp_id` (3-letter residue).
    ATOM_LABEL_NUM = 'atom_label_num'   # Enumerated atoms, mapped from `A_label_atom_id` (atom).
    BB_INDEX = 'bb_index'               # The position of one of the backbone atoms. I've chosen C-alpha ('CA').
    MEAN_COORDS = 'mean_xyz'            # Mean of x y z coordinates for each atom.
    MEAN_CORR_X = 'mean_corrected_x'    # x coordinates for each atom subtracted by the mean of xyz coordinates.
    MEAN_CORR_Y = 'mean_corrected_y'    # (as above) but for y coordinates.
    MEAN_CORR_Z = 'mean_corrected_z'    # (as above) but for z coordinates.


def parse_tokenise_cif_and_write_to_flatfile_to_pdf(pdb_ids=None, use_subdir=False, flatfile: str = 'ssv') -> pd.DataFrame:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `../data/tokenised/`.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    :param use_subdir: True to write the tokenised PDB values to `data` subdir in cwd, otherwise write to the
    larger, general-use, `data` dir that still at top-level of project structure. False by default.
    :param flatfile: Write to ssv, csv or tsv. Use ssv by default.
    :return parsed and tokenised cif file in dataframe, which is also written to ssv in `data/tokenised`.
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        path_to_cif_pdb_ids = f'../diffusion/data/cif/'
        path_to_cif_pdb_ids = path_to_cif_pdb_ids.removesuffix('.cif').removeprefix('/')
        cif = f'{path_to_cif_pdb_ids}{pdb_id}.cif'
        assert os.path.exists(cif)
        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT AND JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, local_cif_file=cif)

        atoms_enumerated, aas_enumerated, fasta_aas_enumerated = dh.read_enumeration_mappings()

        # ENUMERATE BY MAPPING FASTA AMINO ACIDS ('ASP', 'ARG', ETC), USING JSON->DICT `aa_atoms_enumerated`:
        pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')

        # ENUMERATE BY MAPPING ATOM ('C', 'CA', ETC), USING JSON->DICT `aa_atoms_enumerated`:
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')

        # CORRECT ATOMIC X,Y,Z, COORDS BY THEIR MEANS:
        pdf_cif[ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                       CIF.A_Cartn_y.value,
                                                       CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif[ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]

        # REMOVE ORIGINAL NON-ENUMERATED COLUMNS:
        pdf_cif = pdf_cif[[CIF.A_label_asym_id.value,
                           CIF.S_seq_id.value,
                           CIF.A_id.value,
                           CIF.S_mon_id.value,  # temporarily keeping this col but mapped to enumerated form below
                           ColNames.AA_LABEL_NUM.value,  # to replace S_mon_id, but check first it gives what you expect before dropping S_mon_id
                           CIF.A_label_atom_id.value,  # temporarily keeping this col but mapped to enumerated form below
                           ColNames.ATOM_LABEL_NUM.value,  # to replace A_label_atom_id, but check first it gives what you expect before dropping A_label_atom_id
                           ColNames.MEAN_CORR_X.value,
                           ColNames.MEAN_CORR_Y.value,
                           ColNames.MEAN_CORR_Z.value]]
        if flatfile == 'ssv':
            dh.write_tokenised_cif_to_flatfile(pdb_id, pdf_cif, use_subdir=use_subdir, flatfiles='ssv')
        return pdf_cif


if __name__ == '__main__':

    # write_tokenised_cif_to_csv(pdb_ids='4itq')
    parse_tokenise_cif_and_write_to_flatfile_to_pdf(pdb_ids='1OJ6')
    pass
