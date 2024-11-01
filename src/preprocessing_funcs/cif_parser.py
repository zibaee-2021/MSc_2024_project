"""
CIF_PARSER.PY
    - EXTRACT FIELDS OF INTEREST FROM THE RAW MMCIF VIA BIOPYTHON MMCIF2Dict FUNCTIONALITY.
    - MERGE THE TWO DATAFRAMES FROM 2 DISTINCT FIELDS OF THE MMCIF.
    - PARSE:
        - REMOVE HETATM ROWS
        - CAST STRING TYPES TO NUMERIC TYPES
        - REMOVE LOW OCCUPANCY ROWS

--------------------------------------------------------------------------------------------------

(Note the CIF enum includes an `S_` or `A_` prefix, this is just for readability/provenance of each property, so the
strings themselves must be read as a substring from the third character onwards, i.e. A_group_PDB[2:] in order for
Bio.PDB.MMCIF2Dict.MMCIF2Dict(cif) to map to the fields in the raw mmCIF files, which is executed in
`_extract_fields_from_poly_seq(mmcif_dict)` and `_extract_fields_from_atom_site(mmcif_dict)`).

These 14 fields are used and end up in a 14-column dataframe. A description of what they are all used for is given here
and below (I know it's far from ideal duplicating info (even though just comments) but I feel it is beneficial, for now).

A_group_PDB             # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
S_seq_id.value,         # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
S_mon_id.value,         # RESIDUE (3-letter)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
S_pdb_seq_num.value,    # RESIDUE position      - JOIN TO A_auth_seq_id, THEN REMOVE IT.
A_auth_seq_id.value,    # RESIDUE position      - USED TO JOIN WITH S_pdb_seq_num, THEN REMOVE IT.
A_label_comp_id.value,  # RESIDUE (3-letter)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
A_id.value,             # ATOM position         - SORT ON THIS, KEEP IN DF.
A_label_atom_id.value,  # ATOM                  - KEEP IN DF.
A_label_asym_id.value,  # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
S_asym_id.value,        # CHAIN                 - JOIN ON THIS, SORT ON THIS, THEN REMOVE IT.
A_Cartn_x.value,        # ATOM x-coordinates    - X-COORDINATES
A_Cartn_y.value,        # ATOM y-coordinates    - Y-COORDINATES
A_Cartn_z.value,        # ATOM z-coordinates    - Z-COORDINATES
A_occupancy.value       # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api
from src.enums import ColNames, CIF


def _extract_fields_from_poly_seq(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_pdbx_poly_seq_scheme` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    _pdbx_poly_seq_scheme = CIF.S.value                                         # '_pdbx_poly_seq_scheme.'
    seq_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_seq_id.value[2:]]             # RESIDUE POSITION
    mon_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_mon_id.value[2:]]             # RESIDUE (3-LETTER)
    pdb_seq_nums = mmcif[_pdbx_poly_seq_scheme + CIF.S_pdb_seq_num.value[2:]]   # RESIDUE POSITION
    asym_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_asym_id.value[2:]]           # CHAIN

    # 'S_' IS `_pdbx_poly_seq_scheme`
    poly_seq = pd.DataFrame(
        data={
            CIF.S_seq_id.value: seq_ids,                      # 1,1,1,1,1,1,2,2,2,2,2, etc
            CIF.S_mon_id.value: mon_ids,                      # 'ASP', 'ASP', 'ASP', etc
            CIF.S_pdb_seq_num.value: pdb_seq_nums,            # 1,1,1,1,1,1,2,2,2,2,2, etc
            CIF.S_asym_id.value: asym_ids                     # 'A', 'A', 'A', 'A', etc
        })
    return poly_seq


def _extract_fields_from_atom_site(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_atom_site` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    _atom_site = CIF.A.value                                            # '_atom_site.'
    group_pdbs = mmcif[_atom_site + CIF.A_group_PDB.value[2:]]          # GROUP ('ATOM' or 'HETATM')
    ids = mmcif[_atom_site + CIF.A_id.value[2:]]                        # ATOM POSITIONS
    label_atom_ids = mmcif[_atom_site + CIF.A_label_atom_id.value[2:]]  # ATOMS
    label_comp_ids = mmcif[_atom_site + CIF.A_label_comp_id.value[2:]]  # RESIDUE (3-LETTER)
    label_asym_ids = mmcif[_atom_site + CIF.A_label_asym_id.value[2:]]  # CHAIN
    auth_seq_ids = mmcif[_atom_site + CIF.A_auth_seq_id.value[2:]]      # RESIDUE POSITION
    x_coords = mmcif[_atom_site + CIF.A_Cartn_x.value[2:]]              # CARTESIAN X COORDS
    y_coords = mmcif[_atom_site+ CIF.A_Cartn_y.value[2:]]               # CARTESIAN Y COORDS
    z_coords = mmcif[_atom_site + CIF.A_Cartn_z.value[2:]]              # CARTESIAN Z COORDS
    occupancies = mmcif[_atom_site + CIF.A_occupancy.value[2:]]         # OCCUPANCY

    # 'A_' is `_atom_site`
    atom_site = pd.DataFrame(
        data={
            CIF.A_group_PDB.value: group_pdbs,              # 'ATOM' or 'HETATM'
            CIF.A_id.value: ids,                            # 1,2,3,4,5,6,7,8,9,10, etc
            CIF.A_label_atom_id.value: label_atom_ids,      # 'N', 'CA', 'C', 'O', etc
            CIF.A_label_comp_id.value: label_comp_ids,      # 'ASP', 'ASP', 'ASP', etc
            CIF.A_label_asym_id.value: label_asym_ids,      # 'A', 'A', 'A', 'A', etc
            CIF.A_auth_seq_id.value: auth_seq_ids,          # 1,1,1,1,1,1,2,2,2,2,2, etc
            CIF.A_Cartn_x.value: x_coords,                  # COORDS
            CIF.A_Cartn_y.value: y_coords,                  # COORDS
            CIF.A_Cartn_z.value: z_coords,                  # COORDS
            CIF.A_occupancy.value: occupancies              # BETWEEN 0 AND 1.0
        })

    return atom_site


def _wipe_low_occupancy_coords(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    'Occupancy' is the fraction of the atom present at this atom position. Replace all atom coords that have occupancy
    less than or equal to 0.5 with `nan`.
    :param pdf: Pandas dataframe for cif being parsed.
    :return: Given dataframe parsed according to occupancy metric.
    """
    pdf[CIF.A_Cartn_x.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_x.value])
    pdf[CIF.A_Cartn_y.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_y.value])
    pdf[CIF.A_Cartn_z.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_z.value])
    return pdf


def _fetch_mmcif_from_pdb_api_and_write_locally(pdb_id: str) -> None:
    """
    Fetch raw mmCIF data from API (expected hosted at 'https://files.rcsb.org/download/') using given PDB id, and write
    out to flat file.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    """
    response = api.call_for_cif_with_pdb_id(pdb_id)
    mmcif_file = f'../data/big_data_to_git_ignore/cifs_single_domain_prots/{pdb_id}.cif'
    with open(mmcif_file, 'w') as file:
        file.write(response.text)


def _impute_missing_coords(pdf_to_impute, value_to_impute_with=0):
    """
    Impute missing values of the mean x, y, z structure coordinates with 0s.
    :param pdf_to_impute: Dataframe to impute missing data.
    :param value_to_impute_with: Value to use for replace the missing values. Number 0 by default.
    :return: Imputed dataframe.
    """
    pdf_to_impute[[ColNames.MEAN_CORR_X.value,
                   ColNames.MEAN_CORR_Y.value,
                   ColNames.MEAN_CORR_Z.value]] = (pdf_to_impute[[ColNames.MEAN_CORR_X.value,
                                                                  ColNames.MEAN_CORR_Y.value,
                                                                  ColNames.MEAN_CORR_Z.value]].fillna(value_to_impute_with, inplace=False))
    return pdf_to_impute


def parse_cif(pdb_id: str, path_to_raw_cif: str) -> pd.DataFrame:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param path_to_raw_cif: Relative path to locally downloaded raw mmCIF file, (should include the pdb id).
    :return: Necessary fields extracted from raw mmCIF (from local copy or API) and joined in one table.
    """
    path_to_raw_cif = path_to_raw_cif.removesuffix('.cif').removeprefix('/')
    path_to_raw_cif = f'{path_to_raw_cif}.cif'

    if os.path.exists(path_to_raw_cif):
        mmcif = MMCIF2Dict(path_to_raw_cif)
    else:
        print(f'Will try to read {pdb_id} directly from PDB site..')
        _fetch_mmcif_from_pdb_api_and_write_locally(pdb_id)
        mmcif = MMCIF2Dict(path_to_raw_cif)

    poly_seq_fields = _extract_fields_from_poly_seq(mmcif)
    atom_site_fields = _extract_fields_from_atom_site(mmcif)

    # JOIN
    # _atom_site TO `_pdbx_poly_seq_scheme` ON PROTEIN SEQUENCE NUMBER AND CHAIN:
    pdf_merged = pd.merge(
        left=poly_seq_fields,
        right=atom_site_fields,
        left_on=[CIF.S_pdb_seq_num.value, CIF.S_asym_id.value],
        right_on=[CIF.A_auth_seq_id.value, CIF.A_label_asym_id.value],
        how='outer'
    )

    # REMOVE ROWS WITH 'HETATM' GROUP:
    pdf_merged = pdf_merged.drop(pdf_merged[pdf_merged[CIF.A_group_PDB.value] == CIF.HETATM.value].index)
    # ALTERNATIVELY: KEEP ONLY ROWS WITH 'ATOM' GROUP.
    # pdf_merged = pdf_merged[pdf_merged.A_group_PDB == 'ATOM']

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REMOVING 'HETATM' ROWS:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} after removing HETATM rows.')

    # CAST STRINGS OF FLOATS TO NUMERIC:
    # (THE OPERATION BELOW TO REPLACE LOW-OCCUPANCY COORDS WITH NANS WILL FAIL WITHOUT THIS CASTING).
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # CAST STRINGS OF INTS TO NUMERIC AND THEN TO INTEGERS:
    list_of_cols_to_cast = [CIF.S_seq_id.value,         # RESIDUE POSITION
                            CIF.S_pdb_seq_num.value,    # RESIDUE POSITION
                            CIF.A_id.value,             # ATOM POSITION
                            CIF.A_auth_seq_id.value]    # RESIDUE POSITION

    for col_to_cast in [list_of_cols_to_cast]:
        pdf_merged[col_to_cast] = pd.to_numeric(pdf_merged[col_to_cast], errors='coerce')
        pdf_merged[col_to_cast] = pdf_merged[col_to_cast].astype('Int64')

    # REPLACE LOW-OCCUPANCY COORDS WITH NANs:
    pdf_merged = _wipe_low_occupancy_coords(pdf_merged)

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REPLACING LOW-OCCUPANCY COORDS WITH NANs:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after replacing those that have low occupancy with nan.')

    # TODO: NEED TO DECIDE WITH FORMS OF MISSING DATA ARE BETTER TO JUST REMOVE AND WHICH TO IMPUTE WITH 0.
    # IMPUTE MISSING VALUES WITH ZERO
    pdf_merged = _impute_missing_coords(pdf_merged)

    # FILTER OUT ANY ROWS THAT LACK COORDINATES DATA (HERE BASED ON COORDS_X):
    # TODO: NEED TO MAKE SURE THOUGH THAT YOU HAVE AT LEAST ONE BACKBONE ATOM FOR EACH RESIDUE PRESENT IN THE DF.
    #  HENCE IN THOSE CASES, IMPUTE WITH 0 RATHER THAN DELETING THE ROW (I.E. DELETING THAT ATOM).
    pdf_merged = pdf_merged.dropna(subset=[CIF.A_Cartn_x.value], inplace=False)

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REMOVING ROWS WITH NANs IN COORDS_X:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after removing those rows with nan. (Should be 0).')

    # pdf_merged.reset_index(drop=True, inplace=True)

    # READ OUT OF CHAINS (IMPORTANT IF MORE THAN ONE CHAIN) IN SEQUENCE:
    num_of_chains = pdf_merged[CIF.S_asym_id.value].nunique()
    chains = pdf_merged[CIF.S_asym_id.value].unique().tolist()
    print(f'cif with pdb id={pdb_id} has {num_of_chains} chains. \nThey are {chains}.')

    # RE-ORDER COLUMNS:             # WHAT THIS IS          - WHAT I USED IT FOR:
    pdf_merged = pdf_merged[[
        CIF.A_group_PDB.value,      # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
        CIF.S_seq_id.value,         # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
        CIF.S_mon_id.value,         # RESIDUE (3-letter)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
        CIF.S_pdb_seq_num.value,    # RESIDUE position      - JOIN TO A_auth_seq_id, THEN REMOVE IT.
        CIF.A_auth_seq_id.value,    # RESIDUE position      - USED TO JOIN WITH S_pdb_seq_num, THEN REMOVE IT.
        CIF.A_label_comp_id.value,  # RESIDUE (3-letter)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
        CIF.A_id.value,             # ATOM position         - SORT ON THIS, KEEP IN DF.
        CIF.A_label_atom_id.value,  # ATOM                  - KEEP IN DF.
        CIF.A_label_asym_id.value,  # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
        CIF.S_asym_id.value,        # CHAIN                 - JOIN ON THIS, SORT ON THIS, THEN REMOVE IT.
        CIF.A_Cartn_x.value,        # ATOM x-coordinates    - X-COORDINATES
        CIF.A_Cartn_y.value,        # ATOM y-coordinates    - Y-COORDINATES
        CIF.A_Cartn_z.value,        # ATOM z-coordinates    - Z-COORDINATES
        CIF.A_occupancy.value       # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.
    ]]

    # SORT ROWS BY SEQUENCE NUMBERING BY RESIDUE (SEQ ID) THEN BY ATOMS (A_ID) AND THEN CHAIN:
    pdf_merged.reset_index(drop=True, inplace=True)
    pdf_merged = pdf_merged.sort_values([CIF.A_label_asym_id.value, CIF.S_seq_id.value, CIF.A_id.value])

    # ONLY KEEP THESE EIGHT COLUMNS, AND IN THIS ORDER:
    pdf_merged = pdf_merged[[CIF.A_label_asym_id.value,  # CHAIN
                             CIF.S_seq_id.value,         # RESIDUE POSITION
                             CIF.A_id.value,             # ATOM POSITION
                             CIF.S_mon_id.value,         # RESIDUE (3-LETTER)
                             CIF.A_label_atom_id.value,  # ATOM
                             CIF.A_Cartn_x.value,        # X COORDS
                             CIF.A_Cartn_y.value,        # Y COORDS
                             CIF.A_Cartn_z.value]]       # Z COORDS
    return pdf_merged
