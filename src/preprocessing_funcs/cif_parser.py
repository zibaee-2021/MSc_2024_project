"""
CIF_PARSER.PY DOES FOLLOWING:
    - GET THE RAW MMCIF DATA
    - EXTRACT FIELDS OF INTEREST FROM THE RAW MMCIF TO 2 DATAFRAMES
    - REMOVE HETATM ROWS FROM `_atom_site` DATAFRAME
    - MERGE THE 2 DATAFRAMES ON `asym_id` AND `label_asym_id`, THEN ON `seq_id` AND `label_seq_id`
    - CAST STRING TYPES TO NUMERIC TYPES, CAST TEXT (OBJECTS) TO PANDAS STRINGS
    - REMOVE LOW OCCUPANCY ROWS, NANs AND/OR IMPUTE MISSING COORD ROWS
    - SORT BY CERTAIN COLUMNS (`asym_id`, 'seq_id', 'id')
    - REPLACE LOW OCCUPANCY ROWS WITH NANS
    - IMPUTE NANS IN COORDS WITH ZEROS
    - REMOVE UNNECESSARY COLUMNS
--------------------------------------------------------------------------------------------------

(Note the CIF enum includes an `S_` or `A_` prefix, this is just for readability/provenance of each property, so the
strings themselves must be read as a substring from the third character onwards, i.e. A_group_PDB[2:] in order for
Bio.PDB.MMCIF2Dict.MMCIF2Dict(cif) to map to the fields in the raw mmCIF files, which is executed in
`_extract_fields_from_poly_seq(mmcif_dict)` and `_extract_fields_from_atom_site(mmcif_dict)`).

These 14 fields are used and end up in a 14-column dataframe. A description of what they are all used for is given here
and below (I know it's far from ideal duplicating info (even though just comments) but I feel it is beneficial, for now).

'A_' is for `_atom_site`; 'S_' is for `_pdbx_poly_seq_scheme`.

A_group_PDB             # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
S_seq_id.value,         # RESIDUE POSITION      - USED TO JOIN WITH A_label_seq_id. SORT ON THIS, KEEP IN DATAFRAME.
S_mon_id.value,         # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DATAFRAME.
S_pdb_seq_num.value,    # RESIDUE POSITION      - KEEP FOR NOW, AS MAY RELATE TO SEQUENCE AS INPUT TO MAKE EMBEDDINGS.
A_label_seq_id.value,   # RESIDUE POSITION      - USED TO JOIN WITH S_seq_id, THEN REMOVE IT.
A_label_comp_id.value,  # RESIDUE (3-LETTER)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
A_id.value,             # ATOM POSITION         - SORT ON THIS, KEEP IN DATAFRAME.
A_label_atom_id.value,  # ATOM                  - KEEP IN DATAFRAME.
A_label_asym_id.value,  # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DATAFRAME.
S_asym_id.value,        # CHAIN                 - JOIN ON THIS, THEN REMOVE IT.
A_Cartn_x.value,        # COORDINATES           - ATOM X-COORDINATES
A_Cartn_y.value,        # COORDINATES           - ATOM Y-COORDINATES
A_Cartn_z.value,        # COORDINATES           - ATOM Z-COORDINATES
A_occupancy.value       # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.
"""

import os
from typing import List
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api
from src.enums import CIF


def _impute_missing_coords(pdf_to_impute, value_to_impute_with=0):
    """
    Impute missing values of the mean x, y, z structure coordinates with 0s.
    :param pdf_to_impute: Dataframe to impute missing data.
    :param value_to_impute_with: Value to use for replacing missing values with. Number 0 by default.
    :return: Imputed dataframe.
    """
    missing_count = pdf_to_impute[CIF.A_Cartn_x.value].isna().sum()
    print(f'BEFORE imputing, {missing_count} rows with missing values in column {CIF.A_Cartn_x.value}')
    pdf_to_impute[[CIF.A_Cartn_x.value,
                   CIF.A_Cartn_y.value,
                   CIF.A_Cartn_z.value]] = (pdf_to_impute[[CIF.A_Cartn_x.value,
                                                           CIF.A_Cartn_y.value,
                                                           CIF.A_Cartn_z.value]]
                                            .fillna(value_to_impute_with, inplace=False))

    missing_count = pdf_to_impute[CIF.A_Cartn_x.value].isna().sum()
    assert missing_count == 0, (f'AFTER imputing, there should be no rows with missing values, '
                                f'but {missing_count} rows in column {CIF.A_Cartn_x.value} have NANs. '
                                f'Therefore something has gone wrong!')
    return pdf_to_impute


def _replace_low_occupancy_coords_with_nans(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    'Occupancy' is the fraction of the atom present at this atom position. Replace all atom coordinates that have
    occupancy less than or equal to 0.5 with NAN.
    :param pdf: Pandas dataframe for CIF being parsed.
    :return: Given dataframe parsed according to occupancy metric.
    """
    missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    print(f'BEFORE replacing low occupancy rows with NAN, '
          f'there are {missing_count} rows with missing values in column {CIF.A_Cartn_x.value}.')

    pdf[CIF.A_Cartn_x.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_x.value])
    pdf[CIF.A_Cartn_y.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_y.value])
    pdf[CIF.A_Cartn_z.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_z.value])

    missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    print(f'AFTER replacing low occupancy rows with NAN, '
          f'there are {missing_count} rows with missing values in column {CIF.A_Cartn_x.value}.')
    return pdf


def _sort_by_chain_residues_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    # SORT ROWS BY CHAIN, RESIDUE SEQUENCE NUMBERING (SEQ ID) THEN ATOM SEQUENCE NUMBERING (A_ID):
    pdf.reset_index(drop=True, inplace=True)
    pdf = pdf.sort_values([CIF.S_asym_id.value,
                           CIF.S_seq_id.value,
                           CIF.A_id.value])
    return pdf


def _cast_objects_to_stringdtype(pdf: pd.DataFrame) -> pd.DataFrame:
    cols_to_cast = [CIF.S_mon_id.value,
                    CIF.A_label_comp_id.value,
                    CIF.A_label_atom_id.value,
                    CIF.A_label_asym_id.value,
                    CIF.S_asym_id.value]
    for col_to_cast in cols_to_cast:
        pdf[col_to_cast] = pdf[col_to_cast].astype('string')
    return pdf


def _cast_number_strings_to_numeric_types(pdf_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the strings of coordinates which are floats to numeric datatype.
    Cast the strings of integers in `S_seq_id`, `A_label_seq_id`, `S_pdb_seq_num` and `A_id` to numeric, then Int64.
    :param pdf_merged: Data containing columns to cast. It is a dataframe of `_atom_site` and `_pdbx_poly_seq_scheme`
    mmCIF fields merged and with all 'HETATM' rows already removed.
    :return: Data with number strings cast to corresponding numeric types.
    """
    # CAST STRINGS OF FLOATS TO NUMERIC:
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # CAST STRINGS OF INTS TO NUMERIC AND THEN TO INTEGERS:
    list_of_cols_to_cast = [CIF.S_seq_id.value,        # RESIDUE POSITION
                            CIF.A_label_seq_id.value,  # RESIDUE POSITION
                            CIF.S_pdb_seq_num.value,   # RESIDUE POSITION
                            CIF.A_id.value]            # ATOM POSITION
    for col_to_cast in list_of_cols_to_cast:
        pdf_merged[col_to_cast] = pd.to_numeric(pdf_merged[col_to_cast], errors='coerce')
        pdf_merged[col_to_cast] = pdf_merged[col_to_cast].astype('Int64')
    return pdf_merged


def _reorder_columns(pdf_merged: pd.DataFrame) -> pd.DataFrame:
    return pdf_merged[[
        CIF.S_seq_id.value,         # RESIDUE POSITION      - JOIN TO S_label_seq_id, SORT ON THIS, KEEP IN DF.
        CIF.A_label_seq_id.value,   # RESIDUE POSITION      - JOIN TO S_seq_id, THEN REMOVE IT.
        CIF.S_pdb_seq_num.value,    # RESIDUE POSITION      - KEEP FOR NOW, AS MAY RELATE TO INPUT TO MAKE EMBEDDINGS.
        CIF.A_id.value,             # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
        CIF.S_mon_id.value,         # RESIDUE (3-letter)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
        CIF.A_label_comp_id.value,  # RESIDUE (3-letter)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
        CIF.A_label_atom_id.value,  # ATOM                  - KEEP IN DF.
        CIF.A_label_asym_id.value,  # CHAIN                 - JOIN ON THIS, SORT ON THIS, THEN REMOVE IT.
        CIF.S_asym_id.value,        # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
        CIF.A_Cartn_x.value,        # COORDINATES           - X-COORDINATES
        CIF.A_Cartn_y.value,        # COORDINATES           - Y-COORDINATES
        CIF.A_Cartn_z.value,        # COORDINATES           - Z-COORDINATES
        CIF.A_occupancy.value       # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.
    ]]


def _join_atomsite_to_polyseq(atomsite: pd.DataFrame, polyseq: pd.DataFrame) -> pd.DataFrame:
    """
    Join dataframes corresponding to `_atom_site` and `_pdbx_poly_seq_scheme` fields, on protein sequence number.
    :param atomsite:
    :param polyseq:
    :return:
    """
    #
    return pd.merge(
        left=polyseq,
        right=atomsite,
        # left_on=[CIF.S_seq_id.value, CIF.S_asym_id.value],
        left_on=[CIF.S_seq_id.value],
        # right_on=[CIF.A_label_seq_id.value, CIF.A_label_asym_id.value],
        right_on=[CIF.A_label_seq_id.value],
        how='outer'
    )


def _split_up_into_different_chains(atomsite_pdf: pd.DataFrame, polyseq_pdf: pd.DataFrame) -> list:
    """

    :return: List of tuples containing each given pdf for each polypeptide chain,
    e.g. [(`atomsite_pdf_A`, `polyseq_pdf_A`, (`atomsite_pdf_B`, `polyseq_pdf_B`, etc)
    """
    num_of_chains_A = atomsite_pdf[CIF.A_label_asym_id.value].nunique()
    num_of_chains_S = polyseq_pdf[CIF.S_asym_id.value].nunique()
    assert num_of_chains_A == num_of_chains_S, (f'There are {num_of_chains_A} in _atom_site, but {num_of_chains_S} in '
                                                f'_pdbx_poly_seq_scheme!')
    chains = atomsite_pdf[CIF.A_label_asym_id.value].unique()
    grouped_atomsite_dfs = [group_df for _, group_df in atomsite_pdf.groupby(CIF.A_label_asym_id.value)]
    grouped_polyseq_dfs = [group_df for _, group_df in polyseq_pdf.groupby(CIF.S_asym_id.value)]
    grouped_tuple = [(grp_as,grp_ps) for grp_as, grp_ps in zip(grouped_atomsite_dfs, grouped_polyseq_dfs)]
    assert len(chains) == len(grouped_tuple)
    return grouped_tuple


def _remove_hetatm_rows(atomsite_pdf: pd.DataFrame) -> pd.DataFrame:
    missing_count = atomsite_pdf[CIF.A_Cartn_x.value].isna().sum()
    print(f"BEFORE removing 'HETATM' rows, there are {missing_count} rows with missing values in column "
          f"{CIF.A_Cartn_x.value}.")

    atomsite_pdf = atomsite_pdf.drop(atomsite_pdf[atomsite_pdf[CIF.A_group_PDB.value] == CIF.HETATM.value].index)
    # OR KEEP ONLY ROWS WITH 'ATOM' GROUP. NOT SURE IF ONE APPROACH IS BETTER THAN THE OTHER:
    # atom_site_pdf = atom_site_pdf[atom_site_pdf.A_group_PDB == 'ATOM']

    missing_count = atomsite_pdf[CIF.A_Cartn_x.value].isna().sum()
    print(f"AFTER removing 'HETATM' rows, there are {missing_count} rows with missing values in column "
          f"{CIF.A_Cartn_x.value}.")
    return atomsite_pdf


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
    label_seq_ids = mmcif[_atom_site + CIF.A_label_seq_id.value[2:]]      # RESIDUE POSITION
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
            CIF.A_label_seq_id.value: label_seq_ids,          # 1,1,1,1,1,1,2,2,2,2,2, etc
            CIF.A_Cartn_x.value: x_coords,                  # COORDS
            CIF.A_Cartn_y.value: y_coords,                  # COORDS
            CIF.A_Cartn_z.value: z_coords,                  # COORDS
            CIF.A_occupancy.value: occupancies              # BETWEEN 0 AND 1.0
        })

    return atom_site


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


def _get_mmcif_data(pdb_id: str, relpath_to_raw_cif: str) -> dict:
    relpath_to_raw_cif = relpath_to_raw_cif.removesuffix('.cif').removeprefix('/').removesuffix('/')
    relpath_to_raw_cif = f'{relpath_to_raw_cif}/{pdb_id}.cif'

    if os.path.exists(relpath_to_raw_cif):
        mmcif = MMCIF2Dict(relpath_to_raw_cif)
    else:
        print(f'Did not find this CIF locally ({relpath_to_raw_cif}). Attempting to read {pdb_id} directly from '
              f'https://files.rcsb.org/download/{pdb_id}')

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
        _fetch_mmcif_from_pdb_api_and_write_locally(pdb_id)
        mmcif = MMCIF2Dict(relpath_to_raw_cif)
    return mmcif


def parse_cif(pdb_id: str, relpath_to_cifs_dir: str) -> List[pd.DataFrame]:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param relpath_to_cifs_dir: Relative path to local raw mmCIF file, e.g. 'path/to/1OJ6' (MUST INCLUDE PDB ID).
    :return: Necessary fields extracted from raw mmCIF (from local copy or API) and joined in one table.
    """
    mmcif_dict = _get_mmcif_data(pdb_id, relpath_to_cifs_dir)
    polyseq_pdf = _extract_fields_from_poly_seq(mmcif_dict)
    atomsite_pdf = _extract_fields_from_atom_site(mmcif_dict)
    atomsite_pdf = _remove_hetatm_rows(atomsite_pdf)
    chains_pdfs = _split_up_into_different_chains(atomsite_pdf, polyseq_pdf)
    pdfs_merged = []
    for chain_pdfs in chains_pdfs:
        atomsite_pdf, polyseq_pdf = chain_pdfs
        pdf_merged = _join_atomsite_to_polyseq(atomsite_pdf, polyseq_pdf)
        pdf_merged = _reorder_columns(pdf_merged)
        pdf_merged = _cast_number_strings_to_numeric_types(pdf_merged)
        pdf_merged = _cast_objects_to_stringdtype(pdf_merged)
        pdf_merged = _sort_by_chain_residues_atoms(pdf_merged)
        pdf_merged = _replace_low_occupancy_coords_with_nans(pdf_merged)
        pdf_merged = _impute_missing_coords(pdf_merged, value_to_impute_with=0)

        # ONLY KEEP THESE EIGHT COLUMNS, AND IN THIS ORDER:
        pdf_merged = pdf_merged[[CIF.S_asym_id.value,        # CHAIN                * `A_label_asym_id`
                             CIF.S_seq_id.value,         # RESIDUE POSITION     * `A_label_seq_id`
                             CIF.S_mon_id.value,         # RESIDUE              * `A_label_comp`
                             CIF.A_id.value,             # ATOM POSITION        **
                             CIF.A_label_atom_id.value,  # ATOM                 **
                             CIF.A_Cartn_x.value,        # X COORDINATES        ***
                             CIF.A_Cartn_y.value,        # Y COORDINATES        ***
                             CIF.A_Cartn_z.value]]       # Z COORDINATES        ***

    # * THE CORRESPONDING COLUMNS IN `_atom_site` (SHOWN IN LINE ABOVE) MAY HAVE NANS ON SOME ROWS.
    # ** SOME ROWS FROM `_atom_site` MAY HAVE NANS.
    # *** SOME ROWS MAY HAVE HAD NANS, THAT WERE IMPUTED TO 0. THERE SHOULD BE NO NANS IN THESE 3 COLUMNS.
        pdfs_merged.append(pdf_merged)
    return pdfs_merged
