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
    mon_id,             # residue (3-letter)    - used to sanity-check with A_label_comp_id, keep*.
    pdb_seq_num,        # residue position      - join to A_auth_seq_id, then remove.
    asym_id,            # chain                 - join on this, sort on this, then remove.

* After generating enumeration of residues, there's no use for _pdbx_poly_seq_scheme.mon_id, so it can be dropped.
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
    AA_LABEL_NUM = 'aa_label_num'       # Enumerated residues. (Equivalent to `ntcodes` in DJ's RNA code.)
    ATOM_LABEL_NUM = 'atom_label_num'   # Enumerated atoms. (Equivalent to `atomcodes` in DJ's RNA code.)
    BB_INDEX = 'bb_index'               # Position of 1 out the 5 backbone atoms. I've chosen C-alpha ('CA').
    MEAN_COORDS = 'mean_xyz'            # Mean of x y z coordinates for each atom.
    MEAN_CORR_X = 'mean_corrected_x'    # x coordinates for each atom subtracted by the mean of xyz coordinates.
    MEAN_CORR_Y = 'mean_corrected_y'    # (as above) but for y coordinates.
    MEAN_CORR_Z = 'mean_corrected_z'    # (as above) but for z coordinates.


def parse_tokenise_cif_write_flatfile(pdb_ids=None, flatfileformat_to_write: str = 'ssv',
                                      path_to_raw_cifs_dir='data/cif',
                                      dst_path_for_tokenised='data/tokenised') -> pd.DataFrame:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `data/tokenised/`.
    Specifically: enumerate the atoms and residues. Correct x, y, z coordinates by the mean of all 3 per row.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    :param flatfileformat_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param path_to_raw_cifs_dir: Path to source dir of the raw cif files to be parsed and tokenised.
    Use `diffusion/data/cif` subdir by default. Expectation is that this is mostly called from diffusion dir.
    :param dst_path_for_tokenised: Path to destination dir for the parsed and tokenised cif as a flat file.
    Use `diffusion/data/tokenised` subdir by default. Expectation is that this is mostly called from `diffusion` dir.
    If the caller passes in empty string '', it will be interpreted as instruction to write to `diffusion/data/tokenised`.
    :return: Parsed and tokenised cif file in dataframe, which is also written to ssv in `data/tokenised`.
    Dataframe with 13 Columns: 'A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x', 'A_Cartn_y',
    'A_Cartn_z', 'aa_label_num', 'atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y',
    'mean_corrected_z']. NB: 'A_Cartn_x', 'A_Cartn_y', 'A_Cartn_z' and 'mean_xyz' are no longer needed but I'm
    leaving them for sanity-checks (by eye).
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        path_to_raw_cifs_dir = path_to_raw_cifs_dir.removesuffix('.cif').removeprefix('/')
        cif = f'{path_to_raw_cifs_dir}{pdb_id}.cif'
        assert os.path.exists(cif)
        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT AND JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, path_to_raw_cif=cif)
        assert len(pdf_cif.columns) == 8, f'Dataframe should have 8 columns. But this has {len(pdf_cif.columns)}'

        atoms_enumerated, aas_enumerated, fasta_aas_enumerated = dh.read_enumeration_mappings()

        # ENUMERATE BY MAPPING RESIDUES, USING `aa_atoms_enumerated` JSON->DICT:
        pdf_cif.loc[:, ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')
        assert len(pdf_cif.columns) == 9, f'Dataframe should have 9 columns. But this has {len(pdf_cif.columns)}'

        # ENUMERATE BY MAPPING ATOM ('C', 'CA', ETC), USING JSON->DICT `aa_atoms_enumerated`:
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')
        assert len(pdf_cif.columns) == 10, f'Dataframe should have 10 columns. But this has {len(pdf_cif.columns)}'

        # CORRECT ATOMIC X,Y,Z, COORDS BY THEIR MEANS:
        # pdf_cif[ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
        #                                                CIF.A_Cartn_y.value,
        #                                                CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif.loc[:, ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                              CIF.A_Cartn_y.value,
                                                              CIF.A_Cartn_z.value]].mean(axis=1)

        pdf_cif.loc[:, ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]

        # REMOVE ORIGINAL NON-ENUMERATED RESIDUE COLUMN:
        pdf_cif = pdf_cif.drop(columns=[CIF.S_mon_id.value])
        assert len(pdf_cif.columns) == 13, f'Dataframe should have 13 columns. But this has {len(pdf_cif.columns)}'

        dh.write_tokenised_cif_to_flatfile(pdb_id, pdf_cif, dst_data_dir=dst_path_for_tokenised,
                                           flatfiles=flatfileformat_to_write)

        return pdf_cif


if __name__ == '__main__':

    # write_tokenised_cif_to_csv(pdb_ids='4itq')
    print(os.getcwd())
    # Being called from here, which is in the subdir `preprocessing_funcs` so paths must be specified
    parse_tokenise_cif_write_flatfile(pdb_ids='1OJ6', path_to_raw_cifs_dir='../diffusion/data/cif/',
                                      dst_path_for_tokenised='../diffusion/data/tokenised/')
    pass
