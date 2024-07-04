"""
Tokenise each atom.
Tokenise each residue.
Keep track of which residue each atom is associated with.
Take a pdb/mmCIF and tokenise each atom into for example a dict.
The keys of the dict should be:
                                - residue number
                                - atom name
                                    - choose an "anchor atom" (e.g. C3 in RNA, CAlpha or CBeta in protein)
                                    - dict of all atom types
                                        - set of integers
                                        - (Shaun) "enumeration of all characteristics of atom"
                                - atom xyz coordinates

Tokeniser is NOT including coordinates.
Tokeniser "should not be fooled by gaps".
Align atom sequence to PDB sequence.

Ignore any atoms with occupancy less than 0.5. Interpret this as a "missing" atom.
"""
from enum import Enum
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# NOTE: I'm using prefix `S_` for `_pdbx_poly_seq_scheme` and prefix `A_` for `_atom_site`
class CIF(Enum):
    S = '_pdbx_poly_seq_scheme.'
    A = '_atom_site.'

    S_seq_id = 'S_seq_id'
    S_mon_id = 'S_mon_id'
    S_pdb_seq_num = 'S_pdb_seq_num'
    A_group_PDB = 'A_group_PDB'

    A_id = 'A_id'
    A_label_atom_id = 'A_label_atom_id'
    A_label_comp_id = 'A_label_comp_id'
    A_auth_seq_id = 'A_auth_seq_id'
    A_Cartn_x = 'A_Cartn_x'
    A_Cartn_y = 'A_Cartn_y'
    A_Cartn_z = 'A_Cartn_z'
    A_occupancy = 'A_occupancy'


def _extract_fields_from_poly_seq(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_pdbx_poly_seq_scheme` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    seq_ids = mmcif[CIF.S.value + CIF.S_seq_id.value[2:]]
    mon_ids = mmcif[CIF.S.value + CIF.S_mon_id.value[2:]]
    pdb_seq_nums = mmcif[CIF.S.value + CIF.S_pdb_seq_num.value[2:]]

    # 'S_' is `_pdbx_poly_seq_scheme`
    poly_seq = pd.DataFrame(
        data={
            CIF.S_seq_id.value: seq_ids,
            CIF.S_mon_id.value: mon_ids,
            CIF.S_pdb_seq_num.value: pdb_seq_nums,
        })
    return poly_seq


def _extract_fields_from_atom_site(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_atom_site` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    group_pdbs = mmcif[CIF.A.value + CIF.A_group_PDB.value[2:]]
    ids = mmcif[CIF.A.value + CIF.A_id.value[2:]]
    label_atom_ids = mmcif[CIF.A.value + CIF.A_label_atom_id.value[2:]]  #
    label_comp_ids = mmcif[CIF.A.value + CIF.A_label_comp_id.value[2:]]  # aa 3-letter
    auth_seq_ids = mmcif[CIF.A.value + CIF.A_auth_seq_id.value[2:]]
    x_coords = mmcif[CIF.A.value + CIF.A_Cartn_x.value[2:]]
    y_coords = mmcif[CIF.A.value + CIF.A_Cartn_y.value[2:]]
    z_coords = mmcif[CIF.A.value + CIF.A_Cartn_z.value[2:]]
    occupancies = mmcif[CIF.A.value + CIF.A_occupancy.value[2:]]

    # 'A_' is `_atom_site`
    atom_site = pd.DataFrame(
        data={
            CIF.A_group_PDB.value: group_pdbs,
            CIF.A_id.value: ids,
            CIF.A_label_atom_id.value: label_atom_ids,
            CIF.A_label_comp_id.value: label_comp_ids,
            CIF.A_auth_seq_id.value: auth_seq_ids,
            CIF.A_Cartn_x.value: x_coords,
            CIF.A_Cartn_y.value: y_coords,
            CIF.A_Cartn_z.value: z_coords,
            CIF.A_occupancy.value: occupancies
        })

    return atom_site


def _wipe_low_occupancy_coords(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf[CIF.A_Cartn_x.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_x.value])
    pdf[CIF.A_Cartn_y.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_y.value])
    pdf[CIF.A_Cartn_z.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_z.value])
    return pdf


def parse_cif(local_cif_file: str) -> pd.DataFrame:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param local_cif_file: Path to locally downloaded cif file. (Expected in ../data/cifs_csvs directory.)
    :return: Necessary fields extracted and joined in one table.
    """
    mmcif = MMCIF2Dict(local_cif_file)
    poly_seq_fields = _extract_fields_from_poly_seq(mmcif)
    atom_site_fields = _extract_fields_from_atom_site(mmcif)

    pdf_merged = pd.merge(
        left=poly_seq_fields,
        right=atom_site_fields,
        left_on=CIF.S_pdb_seq_num.value,
        right_on=CIF.A_auth_seq_id.value,
        how='outer'
    )

    # pdf_merged.reset_index(drop=True, inplace=True)

    # Cast strings (of floats) to numeric:
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # Cast strings (of ints) to numeric and then to integers:
    for col in [CIF.S_seq_id.value,
                CIF.S_pdb_seq_num.value,
                CIF.A_id.value,
                CIF.A_auth_seq_id.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')
        pdf_merged[col] = pdf_merged[col].astype('Int64')

    # RE-ORDER
    pdf_merged = pdf_merged[[
        CIF.A_group_PDB.value,      # 'ATOM' or 'HETATM'
        CIF.S_seq_id.value,         # amino acid sequence number
        CIF.S_mon_id.value,         # amino acid sequence
        CIF.S_pdb_seq_num.value,    # amino acid sequence number (structure)
        CIF.A_auth_seq_id.value,    # amino acid sequence number (structure)
        CIF.A_label_comp_id.value,  # amino acid sequence (structure)
        # CIF.S_pdb_mon_id.value,     # amino acid sequence (structure)  # not included (redundant ?)
        CIF.A_id.value,             # atom number
        CIF.A_label_atom_id.value,  # atom codes
        CIF.A_Cartn_x.value,        # atom x-coordinates
        CIF.A_Cartn_y.value,        # atom y-coordinates
        CIF.A_Cartn_z.value,        # atom z-coordinates
        CIF.A_occupancy.value       # occupancy
    ]]
    # SORT SEQUENCE NUMBERING BY RESIDUE (SEQ ID) THEN BY ATOMS (A ID):
    pdf_merged = pdf_merged.sort_values([CIF.S_seq_id.value, CIF.A_id.value])
    pdf_merged.reset_index(drop=True, inplace=True)

    # FILTER OUT `HETATM` ROWS:
    # pdf_merged = pdf_merged[pdf_merged.A_group_PDB == 'ATOM']
    pdf_merged = pdf_merged.drop(pdf_merged[pdf_merged['A_group_PDB'] == 'HETATM'].index)

    # REMOVE LOW-OCCUPANCY COORDS:
    pdf_merged = _wipe_low_occupancy_coords(pdf_merged)

    # ONLY KEEP COLUMNS THAT NEEDED:
    pdf_merged = pdf_merged[[CIF.S_seq_id.value,
                             CIF.S_mon_id.value,
                             CIF.A_id.value,
                             CIF.A_label_atom_id.value,
                             CIF.A_Cartn_x.value,
                             CIF.A_Cartn_y.value,
                             CIF.A_Cartn_z.value]]
    return pdf_merged


if __name__ == '__main__':
    pdf_cif = parse_cif(local_cif_file='../data/cifs_csvs/4hb1.cif')
    pdf_cif.to_csv(path_or_buf='../data/cifs_csvs/4hb1_cif.csv', index=False, na_rep='null')
    pdf_cif.to_csv(path_or_buf='../data/cifs_csvs/4hb1_cif.ssv', sep=' ', index=False, na_rep='null')  # space-separated
    pdf_cif.to_csv(path_or_buf='../data/cifs_csvs/4hb1_cif.tsv', sep='\t', index=False, na_rep='null')  # tab-separated
    pdf_easy_read = pdf_cif.rename(columns={CIF.S_seq_id.value: 'SEQ_ID',
                                            CIF.S_mon_id.value: 'RESIDUES',
                                            CIF.A_id.value: 'ATOM_ID',
                                            CIF.A_label_atom_id.value: 'ATOMS',
                                            CIF.A_Cartn_x.value: 'X',
                                            CIF.A_Cartn_y.value: 'Y',
                                            CIF.A_Cartn_z.value: 'Z'})
    pdf_easy_read.to_csv(path_or_buf='../data/cifs_csvs/easyRead_4hb1_cif.tsv', sep='\t', index=False, na_rep='null')

