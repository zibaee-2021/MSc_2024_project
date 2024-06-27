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

# cuda_12.5.r12.5

from enum import Enum
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


if __name__ == '__main__':

    # NOTE: I'm using prefix `S_` for `_pdbx_poly_seq_scheme` and prefix `A_` for `_atom_site`
    class CIF(Enum):
        S = '_pdbx_poly_seq_scheme.'
        A = '_atom_site.'

        S_seq_id = 'S_seq_id'
        S_mon_id = 'S_mon_id'
        S_pdb_seq_num = 'S_pdb_seq_num'
        S_pdb_mon_id = 'S_pdb_mon_id'  # not used as seems to be identical to `A_label_comp_id` (but ? instead of <NA>)
        A_group_PDB = 'A_group_PDB'

        A_id = 'A_id'
        A_label_atom_id = 'A_label_atom_id'
        A_label_comp_id = 'A_label_comp_id'
        A_auth_seq_id = 'A_auth_seq_id'
        A_Cartn_x = 'A_Cartn_x'
        A_Cartn_y = 'A_Cartn_y'
        A_Cartn_z = 'A_Cartn_z'
        A_occupancy = 'A_occupancy'

    cif = '../data/4hb1.cif'
    mmcif_dict = MMCIF2Dict(cif)
    seq_id_list = mmcif_dict[CIF.S.value + CIF.S_seq_id.value[2:]]
    mon_id_list = mmcif_dict[CIF.S.value + CIF.S_mon_id.value[2:]]
    pdb_seq_num_list = mmcif_dict[CIF.S.value + CIF.S_pdb_seq_num.value[2:]]
    pdb_mon_id_list = mmcif_dict[CIF.S.value + CIF.S_pdb_mon_id.value[2:]]

    # 'S_' is `_pdbx_poly_seq_scheme`
    pdf_left = pd.DataFrame(
        data={
            CIF.S_seq_id.value: seq_id_list,
            CIF.S_mon_id.value: mon_id_list,
            CIF.S_pdb_seq_num.value: pdb_seq_num_list,
            CIF.S_pdb_mon_id.value: pdb_mon_id_list
        })

    group_PDB_list = mmcif_dict[CIF.A.value + CIF.A_group_PDB.value[2:]]
    id_list = mmcif_dict[CIF.A.value + CIF.A_id.value[2:]]
    label_atom_id_list = mmcif_dict[CIF.A.value + CIF.A_label_atom_id.value[2:]]  #
    label_comp_id_list = mmcif_dict[CIF.A.value + CIF.A_label_comp_id.value[2:]]  # aa 3-letter
    auth_seq_id_list = mmcif_dict[CIF.A.value + CIF.A_auth_seq_id.value[2:]]
    x_list = mmcif_dict[CIF.A.value + CIF.A_Cartn_x.value[2:]]
    y_list = mmcif_dict[CIF.A.value + CIF.A_Cartn_y.value[2:]]
    z_list = mmcif_dict[CIF.A.value + CIF.A_Cartn_z.value[2:]]
    occupancy_list = mmcif_dict[CIF.A.value + CIF.A_occupancy.value[2:]]

    # 'A_' is `_atom_site`
    pdf_right = pd.DataFrame(
        data={
            CIF.A_group_PDB.value: group_PDB_list,
            CIF.A_id.value: id_list,
            CIF.A_label_atom_id.value: label_atom_id_list,
            CIF.A_label_comp_id.value: label_comp_id_list,
            CIF.A_auth_seq_id.value: auth_seq_id_list,
            CIF.A_Cartn_x.value: x_list,
            CIF.A_Cartn_y.value: y_list,
            CIF.A_Cartn_z.value: z_list,
            CIF.A_occupancy.value: occupancy_list
        })

    pdf_merged = pd.merge(
        left=pdf_left,
        right=pdf_right,
        left_on=CIF.S_pdb_seq_num.value,
        right_on=CIF.A_auth_seq_id.value,
        how='outer'
    )

    # Cast strings (of floats) to numeric
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # Cast strings (of ints) to numeric and then to integers
    for col in [CIF.S_seq_id.value,
                CIF.S_pdb_seq_num.value,
                CIF.A_id.value,
                CIF.A_auth_seq_id.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')
        pdf_merged[col] = pdf_merged[col].astype('Int64')

    cols = pdf_merged.columns
    print(cols)
    col_dtypes = pdf_merged.dtypes
    print(col_dtypes)

    # pdf2.loc[pdf[CIF.occupancy.value] >= 0.5]

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

    # SORT
    pdf_merged = pdf_merged.sort_values([CIF.S_seq_id.value, CIF.A_id.value])

    # FILTER
    # pdf_merged = pdf_merged[pdf_merged.A_group_PDB == 'ATOM']
    pdf_merged = pdf_merged.drop(pdf_merged[pdf_merged['A_group_PDB'] == 'HETATM'].index)
print('yes')


