"""
CIF_PARSER.PY DOES FOLLOWING:
    - GET THE RAW MMCIF DATA
    - EXTRACT FIELDS OF INTEREST FROM THE RAW MMCIF TO 2 DATAFRAMES
    - REMOVE HETATM ROWS FROM `_atom_site` DATAFRAME
    - SEPARATE OUT BY POLYPEPTIDE CHAIN AND PROCESS ONE CHAIN AT A TIME:
        - JOIN THE TWO DATAFRAMES ON RESIDUE POSITION `S_seq_id` AND `A_label_seq_id`
        - CAST STRING TYPES TO NUMERIC TYPES, CAST TEXT (OBJECTS) TO PANDAS STRINGS
        - REMOVE LOW OCCUPANCY ROWS, NANs AND/OR IMPUTE MISSING COORD ROWS
        - SORT BY CERTAIN COLUMN 'S_seq_id' and 'A_id'
        - REPLACE LOW OCCUPANCY ROWS WITH NANS
        - REMOVE ROWS WITH MISSING DATA IN COORDINATES
        - REMOVE UNNECESSARY COLUMNS
--------------------------------------------------------------------------------------------------

(Note the CIF enum includes an `S_` or `A_` prefix, this is just for readability/provenance of each property, so the
strings themselves must be read as a substring from the third character onwards, i.e. A_group_PDB[2:] in order for
Bio.PDB.MMCIF2Dict.MMCIF2Dict(cif) to map to the fields in the raw mmCIF files, which is executed in
`_extract_fields_from_poly_seq(mmcif_dict)` and `_extract_fields_from_atom_site(mmcif_dict)`).

Extract 14 fields from the two joined fields. Output a list of 8-column dataframes, one per chain:

'A_' = `_atom_site`.
'S_' = `_pdbx_poly_seq_scheme`.

A_group_PDB             # 'ATOM' or 'HETATM'    - FILTER ON THIS.                                           THEN REMOVE
S_pdb_seq_num.value,    # RESIDUE POSITION      - (NOT USED)                                                JUST REMOVE
A_label_seq_id.value,   # RESIDUE POSITION      - USED TO JOIN WITH S_seq_id.                               THEN REMOVE
A_label_comp_id.value,  # RESIDUE (3-LETTER)    - USED TO SANITY-CHECK WITH S_mon_id.                       THEN REMOVE
A_label_asym_id.value,  # CHAIN                 - JOIN ON THIS, SORT ON THIS.                               THEN REMOVE
A_occupancy.value       # OCCUPANCY             - FILTER ON THIS.                                           THEN REMOVE

S_asym_id.value,        # CHAIN                 - JOIN ON THIS.                                             THEN KEEP
S_seq_id.value,         # RESIDUE POSITION      - USED TO JOIN WITH A_label_seq_id. SORT ON THIS.           THEN KEEP
S_mon_id.value,         # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id.             THEN KEEP
A_id.value,             # ATOM POSITION         - SORT ON THIS.                                             THEN KEEP
A_label_atom_id.value,  # ATOM                  -                                                           JUST KEEP
A_Cartn_x.value,        # COORDINATES           - ATOM X-COORDINATES                                        JUST KEEP
A_Cartn_y.value,        # COORDINATES           - ATOM Y-COORDINATES                                        JUST KEEP
A_Cartn_z.value,        # COORDINATES           - ATOM Z-COORDINATES                                        JUST KEEP
"""

import os
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api
from src.enums import CIF


class FileExt(Enum):
    dot_CIF = '.cif'


def _remove_rows_with_missing_x_coords(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    :param pdf: IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :return:
    """
    # missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    # if missing_count > 0:
        # print(f'BEFORE removing them, there are {missing_count} rows with missing values in column'
        #       f' {CIF.A_Cartn_x.value}')
    pdf = pdf.dropna(how='any', axis=0, inplace=False, subset=[CIF.A_Cartn_x.value])
    # missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    # if missing_count > 0:
        # print(f'AFTER removing them, there are {missing_count} rows with missing values in column'
        #       f' {CIF.A_Cartn_x.value}')
    return pdf


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


def _process_missing_data(pdf_with_missing_data: pd.DataFrame, impute=False) -> pd.DataFrame:
    """
    Process rows of parsed mmCIF data that having missing data in coordinates (based on `Cartn_x`) by either imputing
    with zeros or removing the rows entirely. Currently, the preferred option is deemed to be the latter,
    as DJ states in Slack answer: "The missing atom threshold is arbitrary - it's just to filter out structures that
    have too many gaps. You could do it by looking at all the data and say rejecting the worst 5%. It doesn't really
    make that much difference. The way the code is currently implemented, there should be no zero or Nan coordinates
    included as padding as the loss function cannot handle them. You could handle this by a mask, but in reality
    there's little point as you get almost the same effect by just skipping them. Just keep track of how many you are
    skipping, though, so that the ntindices are correct. Your approach of correctly filling in the missing sequence
    is probably better - but my approach is good enough for now - given the time, I'd take the same shortcut that I
    took so that you can get something working sooner rather than later.The general idea here is that as long as
    structures are mostly complete, with relatively few not too large gaps then the network just learns to ignore the
    breaks and during inference there will be no breaks as all the atoms will be present and initialized to random
    positions before the denoising process starts... It's one of the benefits of using an atom-level diffusion model
    - it's all just a cloud of labelled atoms."
    :param pdf_with_missing_data: IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param impute:
    :return:
    """
    if impute:
        result_pdf = _impute_missing_coords(pdf_with_missing_data)
    else:
        result_pdf = _remove_rows_with_missing_x_coords(pdf_with_missing_data)
    return result_pdf


def _replace_low_occupancy_coords_with_nans(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    'Occupancy' is the fraction of the atom present at this atom position. Replace all atom coordinates that have
    occupancy less than or equal to 0.5 with NAN.
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param pdf: Pandas dataframe for CIF being parsed.
    :return: Given dataframe parsed according to occupancy metric.
    """
    missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    # if missing_count > 0:
        # print(f'BEFORE replacing low occupancy rows with NAN, '
        #       f'there are {missing_count} rows with missing values in column {CIF.A_Cartn_x.value}.')

    pdf[CIF.A_Cartn_x.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_x.value])
    pdf[CIF.A_Cartn_y.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_y.value])
    pdf[CIF.A_Cartn_z.value] = np.where(pdf[CIF.A_occupancy.value] <= 0.5, np.nan, pdf[CIF.A_Cartn_z.value])

    missing_count = pdf[CIF.A_Cartn_x.value].isna().sum()
    # if missing_count > 0:
    #     print(f'AFTER replacing low occupancy rows with NAN, '
              # f'there are {missing_count} rows with missing values in column {CIF.A_Cartn_x.value}.')
    return pdf


def _sort_by_chain_residues_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param pdf:
    :return:
    """
    # SORT ROWS BY CHAIN, RESIDUE SEQUENCE NUMBERING (SEQ ID) THEN ATOM SEQUENCE NUMBERING (A_ID):
    pdf.reset_index(drop=True, inplace=True)
    pdf = pdf.sort_values([CIF.S_seq_id.value, CIF.A_id.value], ignore_index=True)
    return pdf


def _cast_objects_to_stringdtype(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param pdf:
    :return:
    """
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
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    Cast the strings of integers in `S_seq_id`, `A_label_seq_id`, `S_pdb_seq_num` and `A_id` to numeric, then Int64.
    :param pdf_merged: Data containing columns to cast. It is a dataframe of `_atom_site` and `_pdbx_poly_seq_scheme`
    mmCIF fields merged, and with all 'HETATM' rows already removed.
    :return: Data with number strings cast to corresponding numeric types.
    """
    # CAST STRINGS OF FLOATS TO NUMERIC:
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged.loc[:, col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # CAST STRINGS OF INTS TO NUMERIC AND THEN TO INTEGERS:
    list_of_cols_to_cast = [CIF.S_seq_id.value,        # RESIDUE POSITION
                            CIF.A_label_seq_id.value,  # RESIDUE POSITION
                            CIF.S_pdb_seq_num.value,   # RESIDUE POSITION
                            CIF.A_id.value]            # ATOM POSITION
    for col_to_cast in list_of_cols_to_cast:
        pdf_merged.loc[:, col_to_cast] = pd.to_numeric(pdf_merged[col_to_cast], errors='coerce')
        pdf_merged.loc[:, col_to_cast] = pdf_merged[col_to_cast].astype('Int64')
    return pdf_merged


def _reorder_columns(pdf_merged: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param pdf_merged:
    :return:
    """
    return pdf_merged[[
        CIF.S_seq_id.value,         # RESIDUE POSITION      - JOIN TO S_label_seq_id, SORT ON THIS, KEEP IN DF.
        CIF.A_label_seq_id.value,   # RESIDUE POSITION      - JOIN TO S_seq_id, THEN REMOVE IT.
        CIF.S_pdb_seq_num.value,    # RESIDUE POSITION      - KEEP FOR NOW, AS MAY RELATE TO INPUT TO MAKE EMBEDDINGS.
        CIF.A_id.value,             # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
        CIF.S_mon_id.value,         # RESIDUE (3-letter)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
        CIF.A_label_comp_id.value,  # RESIDUE (3-letter)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
        CIF.A_label_atom_id.value,  # ATOM                  - KEEP IN DF.
        CIF.A_label_asym_id.value,  # CHAIN                 - JUST REMOVE
        CIF.S_asym_id.value,        # CHAIN                 - JUST KEEP
        CIF.A_Cartn_x.value,        # COORDINATES           - X-COORDINATES
        CIF.A_Cartn_y.value,        # COORDINATES           - Y-COORDINATES
        CIF.A_Cartn_z.value,        # COORDINATES           - Z-COORDINATES
        CIF.A_occupancy.value       # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.
    ]]


def _join_atomsite_to_polyseq(atomsite: pd.DataFrame, polyseq: pd.DataFrame) -> pd.DataFrame:
    """
    Join dataframes corresponding to `_atom_site` and `_pdbx_poly_seq_scheme` fields, on protein sequence number.
    IMPORTANT: THIS IS EXPECTED TO BE FOR ON POLYPEPTIDE CHAIN ONLY.
    :param atomsite:
    :param polyseq:
    :return:
    """
    #
    return pd.merge(
        left=polyseq,
        right=atomsite,
        left_on=[CIF.S_seq_id.value],
        right_on=[CIF.A_label_seq_id.value],
        how='outer'
    )


def _split_up_by_chain(atomsite_pdf: pd.DataFrame, polyseq_pdf: pd.DataFrame) -> list:
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
    if missing_count > 0:
        print(f"BEFORE removing 'HETATM' rows, there are {missing_count} rows with missing values in column "
              f"{CIF.A_Cartn_x.value}.")

    atomsite_pdf = atomsite_pdf.drop(atomsite_pdf[atomsite_pdf[CIF.A_group_PDB.value] == CIF.HETATM.value].index)
    # OR KEEP ONLY ROWS WITH 'ATOM' GROUP. NOT SURE IF ONE APPROACH IS BETTER THAN THE OTHER:
    # atom_site_pdf = atom_site_pdf[atom_site_pdf.A_group_PDB == 'ATOM']

    missing_count = atomsite_pdf[CIF.A_Cartn_x.value].isna().sum()
    if missing_count > 0:
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
    label_seq_ids = mmcif[_atom_site + CIF.A_label_seq_id.value[2:]]    # RESIDUE POSITION
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


def __fetch_mmcif_from_pdb_api_and_write(_pdb_id: str, relpath_dst_cif: str) -> None:
    """
    Fetch raw mmCIF data from API (expected hosted at 'https://files.rcsb.org/download/') using given PDB id,
    and write out to flat file.
    :param _pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    """
    response = api.call_for_cif_with_pdb_id(_pdb_id)
    # relpath_dst_cif = f'../data/big_data_to_git_ignore/SD_573_CIFs/{pdb_id}.cif'
    with open(relpath_dst_cif, 'w') as file:
        file.write(response.text)


def _get_mmcif_data(pdb_id: str, relpath_raw_cif: str) -> dict:
    relpath_raw_cif = relpath_raw_cif.removesuffix(FileExt.dot_CIF.value).removeprefix('/').removesuffix('/')
    pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
    relpath_raw_cif = f'{relpath_raw_cif}/{pdb_id}{FileExt.dot_CIF.value}'
    if os.path.exists(relpath_raw_cif):
        try:
            MMCIF2Dict(relpath_raw_cif)
        except ValueError:
            print(f'{pdb_id}{FileExt.dot_CIF.value} appears to be empty. '
                  f'Attempt to read {pdb_id} directly from https://files.rcsb.org/download/{pdb_id}')
            __fetch_mmcif_from_pdb_api_and_write(_pdb_id=pdb_id, relpath_dst_cif=relpath_raw_cif)
    else:
        print(f'Did not find this CIF locally ({relpath_raw_cif}). '
              f'Attempt to read {pdb_id} directly from https://files.rcsb.org/download/{pdb_id}')
        __fetch_mmcif_from_pdb_api_and_write(_pdb_id=pdb_id, relpath_dst_cif=relpath_raw_cif)
    mmcif = MMCIF2Dict(relpath_raw_cif)
    return mmcif


def parse_cif(pdb_id: str, relpath_cifs_dir: str) -> List[pd.DataFrame]:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param relpath_cifs_dir: Relative path to local raw mmCIF file, e.g. 'path/to/1OJ6' (MUST INCLUDE PDB ID).
    :return: 8 necessary fields extracted from raw mmCIF (from local copy or API) and joined in one table.
    This is a list of one or more results for each chain found in this mmCIF.
    """
    print(f'Start parsing PDBid={pdb_id}')
    mmcif_dict = _get_mmcif_data(pdb_id, relpath_cifs_dir)
    polyseq_pdf = _extract_fields_from_poly_seq(mmcif_dict)
    atomsite_pdf = _extract_fields_from_atom_site(mmcif_dict)
    atomsite_pdf = _remove_hetatm_rows(atomsite_pdf)
    # GENERATE A LIST OF TUPLES, EACH TUPLE IS THE ATOMSITE AND POLYSEQ DATA FOR A SINGLE CHAIN
    all_chains_pdf = _split_up_by_chain(atomsite_pdf, polyseq_pdf)

    parsed_cif_by_chain = []
    for chain_pdf in all_chains_pdf:
        atomsite_pdf, polyseq_pdf = chain_pdf
        joined_pdf = _join_atomsite_to_polyseq(atomsite_pdf, polyseq_pdf)
        joined_pdf = _reorder_columns(joined_pdf)
        joined_pdf = _cast_number_strings_to_numeric_types(joined_pdf)
        joined_pdf = _cast_objects_to_stringdtype(joined_pdf)
        joined_pdf = _sort_by_chain_residues_atoms(joined_pdf)
        joined_pdf = _replace_low_occupancy_coords_with_nans(joined_pdf)
        joined_pdf = _process_missing_data(joined_pdf, impute=False)

        # 8 COLUMNS TO KEEP:
        joined_pdf = joined_pdf[[CIF.S_asym_id.value,        # CHAIN
                                 CIF.S_seq_id.value,         # RESIDUE POSITION
                                 CIF.S_mon_id.value,         # RESIDUE NAME (3-LETTER)
                                 CIF.A_id.value,             # ATOM POSITION
                                 CIF.A_label_atom_id.value,  # ATOM NAME
                                 CIF.A_Cartn_x.value,
                                 CIF.A_Cartn_y.value,
                                 CIF.A_Cartn_z.value]]
        parsed_cif_by_chain.append(joined_pdf)
    return parsed_cif_by_chain
