"""
Tokenise each atom.
Tokenise each residue.
Keep track of which residue each atom is associated with.
Take a pdb/mmCIF and tokenise each atom into for example a dict.
The keys of the dict should be:
                                - residue number
                                - atom name
                                    - choose an "anchor atom" (e.g. C3 in RNA; CAlpha or CBeta in protein)
                                    - dict of all atom types
                                        - set of integers
                                        - (Shaun: "enumeration of all characteristics of atom")
                                - atom xyz coordinates

Tokeniser is NOT including coordinates.
Tokeniser "should not be fooled by gaps".
Align atom sequence to PDB sequence.

Ignore any atoms with occupancy less than 0.5. Interpret this as a "missing" atom.

--------------------------------------------------------------------------------------------------
What some of the sequence labels/properties (https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Items/) are is not
so obvious from their names, so I list and describe them here:

_pdbx_poly_seq_scheme
    .seq_id             Pointer to _atom_site.label_seq_id, itself a pointer to _entity_poly_seq.num:
                                Sequence number, must be unique and increasing.
    .mon_id             Pointer to _entity_poly_seq.mon_id, itself a pointer to _chem_comp.id:
                                3-letter amino acid code, or 1-letter nucleic acid base code.
    .pdb_seq_num        PDB residue number.
    .asym_id            Pointer to _atom_site.label_asym_id (PDB chain identifier).

_atom_site
    .group_PDB          Placeholder for tags used by PDB to identify coordinate records (e.g 'ATOM' or 'HETATM').
    .id                 A unique identifier for each atom position (here is a number).
    .label_atom_id      PDB atom identifier (here is a name string, 'C', 'CA', etc).
    .label_comp_id      PDB 3-letter-code residue names. SANITY-CHECK: DO ATOM_SITE & PDBX_POLY_SEQ_SCHEME GIVE SAME AA
    .label_asym_id      PDB chain identifier.
    .auth_seq_id        PDB residue number. (Author defined alternative to _atom_site.label_seq_id).
    .Cartn_x            Cartesian X coordinate component describing the position of this atom site.
    .Cartn_y            Cartesian Y coordinate component describing the position of this atom site.
    .Cartn_z            Cartesian Z coordinate component describing the position of this atom site.
    .occupancy          The fraction of the atom present at this atom position.

(Note the CIF enum includes an `S_` or `A_` prefix, this is just for readability/provenance of each property, so the
strings themselves are always only read as a substring from the third character onwards.)

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
from enum import Enum
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api


# NOTE: I'm using prefix `S_` for `_pdbx_poly_seq_scheme` and prefix `A_` for `_atom_site`
# These prefixes were added to help keep track of the provenance of the CIF field names up to the point of joining
# dataframes.

class CIF(Enum):
    S = '_pdbx_poly_seq_scheme.'
    A = '_atom_site.'

    S_seq_id = 'S_seq_id'                   # residue position*
    S_mon_id = 'S_mon_id'                   # residue (3-letter)
    S_pdb_seq_num = 'S_pdb_seq_num'         # residue position*
    S_asym_id = 'S_asym_id'                 # chain

    A_group_PDB = 'A_group_PDB'             # group, ('ATOM' or 'HETATM')
    A_id = 'A_id'                           # atom position*
    A_label_atom_id = 'A_label_atom_id'     # atom
    A_label_comp_id = 'A_label_comp_id'     # residue (3-letter)
    A_label_asym_id = 'A_label_asym_id'     # chain
    A_auth_seq_id = 'A_auth_seq_id'         # residue position*
    A_Cartn_x = 'A_Cartn_x'
    A_Cartn_y = 'A_Cartn_y'
    A_Cartn_z = 'A_Cartn_z'
    A_occupancy = 'A_occupancy'

    HETATM = 'HETATM'
# * All of these position indices are never less than 1 (i.e. never 0).


def _extract_fields_from_poly_seq(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_pdbx_poly_seq_scheme` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    _pdbx_poly_seq_scheme = CIF.S.value                                         # '_pdbx_poly_seq_scheme.'
    seq_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_seq_id.value[2:]]             # residue position
    mon_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_mon_id.value[2:]]             # residue (3-letter)
    pdb_seq_nums = mmcif[_pdbx_poly_seq_scheme + CIF.S_pdb_seq_num.value[2:]]   # residue position
    asym_ids = mmcif[_pdbx_poly_seq_scheme + CIF.S_asym_id.value[2:]]           # chain

    # 'S_' is `_pdbx_poly_seq_scheme`
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
    group_pdbs = mmcif[_atom_site + CIF.A_group_PDB.value[2:]]          # group ('ATOM' or 'HETATM')
    ids = mmcif[_atom_site + CIF.A_id.value[2:]]                        # atom positions
    label_atom_ids = mmcif[_atom_site + CIF.A_label_atom_id.value[2:]]  # atoms
    label_comp_ids = mmcif[_atom_site + CIF.A_label_comp_id.value[2:]]  # residue (3-letter)
    label_asym_ids = mmcif[_atom_site + CIF.A_label_asym_id.value[2:]]  # chain
    auth_seq_ids = mmcif[_atom_site + CIF.A_auth_seq_id.value[2:]]      # residue number.
    x_coords = mmcif[_atom_site + CIF.A_Cartn_x.value[2:]]              # Cartesian X coord
    y_coords = mmcif[_atom_site+ CIF.A_Cartn_y.value[2:]]               # Cartesian Y coord
    z_coords = mmcif[_atom_site + CIF.A_Cartn_z.value[2:]]              # Cartesian Z coord
    occupancies = mmcif[_atom_site + CIF.A_occupancy.value[2:]]         # Fraction of atom present at this position.

    # 'A_' is `_atom_site`
    atom_site = pd.DataFrame(
        data={
            CIF.A_group_PDB.value: group_pdbs,              # 'ATOM' or 'HETATM'
            CIF.A_id.value: ids,                            # 1,2,3,4,5,6,7,8,9,10, etc
            CIF.A_label_atom_id.value: label_atom_ids,      # 'N', 'CA', 'C', 'O', etc
            CIF.A_label_comp_id.value: label_comp_ids,      # 'ASP', 'ASP', 'ASP', etc
            CIF.A_label_asym_id.value: label_asym_ids,      # 'A', 'A', 'A', 'A', etc
            CIF.A_auth_seq_id.value: auth_seq_ids,          # 1,1,1,1,1,1,2,2,2,2,2, etc
            CIF.A_Cartn_x.value: x_coords,                  # coords
            CIF.A_Cartn_y.value: y_coords,                  # coords
            CIF.A_Cartn_z.value: z_coords,                  # coords
            CIF.A_occupancy.value: occupancies              # between 0 and 1.0
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


def parse_cif(pdb_id: str, path_to_raw_cif: str) -> pd.DataFrame:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param path_to_raw_cif: Relative path to locally downloaded raw mmCIF file, (should include the pdb id).
    :return: Necessary fields extracted from raw mmCIF (from local copy or API) and joined in one table.
    """
    # `os.path.exists` expected no leading fwd slash
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
    # Notice: I've left the prefixes of 'A_' and 'S_' on these column names, (not necessary but leaving as is for now.)
    pdf_merged = pd.merge(
        left=poly_seq_fields,
        right=atom_site_fields,
        left_on=[CIF.S_pdb_seq_num.value, CIF.S_asym_id.value],
        right_on=[CIF.A_auth_seq_id.value, CIF.A_label_asym_id.value],
        how='outer'
    )

    # FILTER OUT `HETATM` ROWS:
    pdf_merged = pdf_merged.drop(pdf_merged[pdf_merged[CIF.A_group_PDB.value] == CIF.HETATM.value].index)
    # pdf_merged = pdf_merged[pdf_merged.A_group_PDB == 'ATOM']  # Alternative: only keep rows starting 'ATOM'

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REMOVING 'HETATM' ROWS:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} after removing HETATM rows.')

    # CAST STRINGS OF FLOATS TO NUMERIC:
    # (LOW-OCCUPANCY OPERATION BELOW WILL FAIL WITHOUT THIS)
    for col in [CIF.A_Cartn_x.value,
                CIF.A_Cartn_y.value,
                CIF.A_Cartn_z.value,
                CIF.A_occupancy.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    # CAST STRINGS OF INTS TO NUMERIC AND THEN TO INTEGERS:
    for col in [CIF.S_seq_id.value,         # residue position
                CIF.S_pdb_seq_num.value,    # residue position
                CIF.A_id.value,             # atom position
                CIF.A_auth_seq_id.value]:   # residue position
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')
        pdf_merged[col] = pdf_merged[col].astype('Int64')

    # REPLACE LOW-OCCUPANCY COORDS WITH NANs:
    pdf_merged = _wipe_low_occupancy_coords(pdf_merged)

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REPLACING LOW OCCUPANCY COORDS WITH NANS:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after replacing those that have low occupancy with nan.')

    # FILTER OUT ANY ROWS THAT LACK COORDINATES DATA (HERE BASED ON COORDS_X):
    pdf_merged = pdf_merged.dropna(subset=[CIF.A_Cartn_x.value], inplace=False)

    # COUNT ROWS WITH MISSING VALUES IN COORDS_X, AFTER REMOVING ROWS WITH NANS IN COORDS_X:
    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after removing those rows with nan. (Should be 0).')

    # pdf_merged.reset_index(drop=True, inplace=True)

    # READ OUT OF CHAINS (IMPORTANT IF MORE THAN ONE CHAIN) IN SEQUENCE:
    num_of_chains = pdf_merged[CIF.S_asym_id.value].nunique()
    chains = pdf_merged[CIF.S_asym_id.value].unique().tolist()
    print(f'cif with pdb id={pdb_id} has {num_of_chains} chains. \nThey are {chains}.')

    # RE-ORDER COLUMNS:
    pdf_merged = pdf_merged[[
        CIF.A_group_PDB.value,      # 'ATOM' or 'HETATM'    - Filter on this then remove.
        CIF.S_seq_id.value,         # residue position      - sort on this, keep.
        CIF.S_mon_id.value,         # residue (3-letter)    - used to sanity-check with A_label_comp_id, keep.
        CIF.S_pdb_seq_num.value,    # residue position      - join to A_auth_seq_id, then remove.
        CIF.A_auth_seq_id.value,    # residue position      - used to join with S_pdb_seq_num, then remove.
        CIF.A_label_comp_id.value,  # residue (3-letter)    - used to sanity-check with S_mon_id, then remove.
        CIF.A_id.value,             # atom position         - sort on this, keep.
        CIF.A_label_atom_id.value,  # atom                  - keep
        CIF.A_label_asym_id.value,  # chain                 - join on this, sort on this, keep.
        CIF.S_asym_id.value,        # chain                 - join on this, sort on this, then remove.
        CIF.A_Cartn_x.value,        # atom x-coordinates
        CIF.A_Cartn_y.value,        # atom y-coordinates
        CIF.A_Cartn_z.value,        # atom z-coordinates
        CIF.A_occupancy.value       # occupancy
    ]]

    # SORT ROWS BY SEQUENCE NUMBERING BY RESIDUE (SEQ ID) THEN BY ATOMS (A_ID) AND THEN CHAIN:
    pdf_merged.reset_index(drop=True, inplace=True)
    pdf_merged = pdf_merged.sort_values([CIF.A_label_asym_id.value, CIF.S_seq_id.value, CIF.A_id.value])

    # ONLY KEEP THESE EIGHT COLUMNS, AND IN THIS ORDER:
    pdf_merged = pdf_merged[[CIF.A_label_asym_id.value,  # chain
                             CIF.S_seq_id.value,         # residue position
                             CIF.A_id.value,             # atom position
                             CIF.S_mon_id.value,         # residue (3-letter)
                             CIF.A_label_atom_id.value,  # atom
                             CIF.A_Cartn_x.value,        # x
                             CIF.A_Cartn_y.value,        # y
                             CIF.A_Cartn_z.value]]       # z
    return pdf_merged
