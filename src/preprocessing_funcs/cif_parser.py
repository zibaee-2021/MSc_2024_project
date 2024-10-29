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
    .label_comp_id      PDB 3-letter-code residue names.
    .label_asym_id      PDB chain identifier.
    .auth_seq_id        PDB residue number. (Author defined alternative to _atom_site.label_seq_id).
    .Cartn_x            Cartesian X coordinate component describing the position of this atom site.
    .Cartn_y            Cartesian Y coordinate component describing the position of this atom site.
    .Cartn_z            Cartesian Z coordinate component describing the position of this atom site.
    .occupancy          The fraction of the atom present at this atom position.

(Note the CIF enum includes an `S_` or `A_` prefix, this is just for readability/provenance of each property, so the
strings themselves are always only read as a substring from the third character onwards.)
"""
import os
from enum import Enum
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api


# NOTE: I'm using prefix `S_` for `_pdbx_poly_seq_scheme` and prefix `A_` for `_atom_site`
class CIF(Enum):
    S = '_pdbx_poly_seq_scheme.'
    A = '_atom_site.'

    S_seq_id = 'S_seq_id'
    S_mon_id = 'S_mon_id'
    S_pdb_seq_num = 'S_pdb_seq_num'
    S_asym_id = 'S_asym_id'
    A_group_PDB = 'A_group_PDB'

    A_id = 'A_id'
    A_label_atom_id = 'A_label_atom_id'
    A_label_comp_id = 'A_label_comp_id'
    A_label_asym_id = 'A_label_asym_id'
    A_auth_seq_id = 'A_auth_seq_id'
    A_Cartn_x = 'A_Cartn_x'
    A_Cartn_y = 'A_Cartn_y'
    A_Cartn_z = 'A_Cartn_z'
    A_occupancy = 'A_occupancy'

    HETATM = 'HETATM'


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
    asym_ids = mmcif[CIF.S.value + CIF.S_asym_id.value[2:]]

    # 'S_' is `_pdbx_poly_seq_scheme`
    poly_seq = pd.DataFrame(
        data={
            CIF.S_seq_id.value: seq_ids,
            CIF.S_mon_id.value: mon_ids,
            CIF.S_pdb_seq_num.value: pdb_seq_nums,
            CIF.S_asym_id.value: asym_ids
        })
    return poly_seq


def _extract_fields_from_atom_site(mmcif: dict) -> pd.DataFrame:
    """
    Extract necessary fields from `_atom_site` records from the given mmCIF (expected as dict).
    (One or more fields might not be necessary for subsequent tokenisation but are not yet removed).
    :param mmcif:
    :return: mmCIF fields in tabulated format.
    """
    group_pdbs = mmcif[CIF.A.value + CIF.A_group_PDB.value[2:]]  # (e.g 'ATOM' or 'HETATM')
    ids = mmcif[CIF.A.value + CIF.A_id.value[2:]]  # '_atom_site.' + 'A_id'  atom index position
    label_atom_ids = mmcif[CIF.A.value + CIF.A_label_atom_id.value[2:]]  # PDB atom name.
    label_comp_ids = mmcif[CIF.A.value + CIF.A_label_comp_id.value[2:]]  # PDB 3-letter-code residue names.
    label_asym_ids = mmcif[CIF.A.value + CIF.A_label_asym_id.value[2:]]  # PDB chain identifier.
    auth_seq_ids = mmcif[CIF.A.value + CIF.A_auth_seq_id.value[2:]]  # PDB residue number.
    x_coords = mmcif[CIF.A.value + CIF.A_Cartn_x.value[2:]]  # Cartesian X coord
    y_coords = mmcif[CIF.A.value + CIF.A_Cartn_y.value[2:]]  # Cartesian Y coord
    z_coords = mmcif[CIF.A.value + CIF.A_Cartn_z.value[2:]]  # Cartesian Z coord
    occupancies = mmcif[CIF.A.value + CIF.A_occupancy.value[2:]]  # Fraction of atom present at this atom position.

    # 'A_' is `_atom_site`
    atom_site = pd.DataFrame(
        data={
            CIF.A_group_PDB.value: group_pdbs,              # 'ATOM' or 'HETATM'
            CIF.A_id.value: ids,                            # e.g. 0, 1000, 10000
            CIF.A_label_atom_id.value: label_atom_ids,      # e.g. 'C', 'CA', etc
            CIF.A_label_comp_id.value: label_comp_ids,      # e.g. 'ASP', 'ARG', etc
            CIF.A_label_asym_id.value: label_asym_ids,      # e.g. 'A', 'B', etc
            CIF.A_auth_seq_id.value: auth_seq_ids,          # residue index position
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
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1oj6'.
    """
    response = api.call_for_cif_with_pdb_id(pdb_id)
    mmcif_file = f'../data/big_data_to_git_ignore/cifs_single_domain_prots/{pdb_id}.cif'
    with open(mmcif_file, 'w') as file:
        file.write(response.text)


def parse_cif(pdb_id: str, local_cif_file: str) -> pd.DataFrame:
    """
    Parse given local mmCIF file to extract and tabulate necessary atom and amino acid data fields from
    `_pdbx_poly_seq_scheme` and `_atom_site`.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param local_cif_file: Relative path to locally downloaded mmCIF file.
    :return: Necessary fields extracted from raw mmCIF (from local copy or API) and joined in one table.
    """
    local_cif_file = local_cif_file.removeprefix('/')  # `os.path.exists` expected no leading fwd slash
    if os.path.exists(local_cif_file):
        mmcif = MMCIF2Dict(local_cif_file)
    else:
        print(f'Will try to read {pdb_id} directly from PDB site..')
        _fetch_mmcif_from_pdb_api_and_write_locally(pdb_id)
        mmcif = MMCIF2Dict(local_cif_file)

    poly_seq_fields = _extract_fields_from_poly_seq(mmcif)
    atom_site_fields = _extract_fields_from_atom_site(mmcif)

    # JOIN _atom_site TO `_pdbx_poly_seq_scheme` ON PROTEIN SEQUENCE NUMBER AND CHAIN:
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
    for col in [CIF.S_seq_id.value,
                CIF.S_pdb_seq_num.value,
                CIF.A_id.value,
                CIF.A_auth_seq_id.value]:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')
        pdf_merged[col] = pdf_merged[col].astype('Int64')

    # REPLACE LOW-OCCUPANCY COORDS WITH NANs:
    pdf_merged = _wipe_low_occupancy_coords(pdf_merged)

    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after replacing those that have low occupancy with nan.')

    # FILTER OUT 'NAN' ROWS
    pdf_merged = pdf_merged.dropna(subset=[CIF.A_Cartn_x.value], inplace=False)

    missing_count = pdf_merged[CIF.A_Cartn_x.value].isna().sum()
    print(f'{missing_count} rows have missing values in column {CIF.A_Cartn_x.value} '
          f'after removing those rows with nan. (Should be 0).')

    # pdf_merged.reset_index(drop=True, inplace=True)

    # READ OUT OF CHAINS (IMPORTANT IF MORE THAN ONE CHAIN) IN SEQUENCE:
    num_of_chains = pdf_merged[CIF.S_asym_id.value].nunique()
    chains = pdf_merged[CIF.S_asym_id.value].unique().tolist()
    print(f'cif with pdb id={pdb_id} has {num_of_chains} chains. \nThey are {chains}.')

    # RE-ORDER COLUMNS
    pdf_merged = pdf_merged[[
        CIF.A_group_PDB.value,      # 'ATOM' or 'HETATM'
        CIF.S_seq_id.value,         # amino acid sequence number
        CIF.S_mon_id.value,         # amino acid sequence (3-letter)
        CIF.S_pdb_seq_num.value,    # amino acid sequence number (structure) - duplicating S_seq_id
        CIF.A_auth_seq_id.value,    # amino acid sequence number (structure) - duplicating S_seq_id
        CIF.A_label_comp_id.value,  # amino acid sequence (structure) - duplicating S_mon_id
        CIF.A_id.value,             # atom identifier to indicate position (happens to be a number)
        CIF.A_label_atom_id.value,  # atom identifier (happens to be a name string)
        CIF.A_label_asym_id.value,  # atom chain - this is the polypeptide chain
        CIF.S_asym_id.value,        # polypeptide chain - duplicating A_label_asym_id
        CIF.A_Cartn_x.value,        # atom x-coordinates
        CIF.A_Cartn_y.value,        # atom y-coordinates
        CIF.A_Cartn_z.value,        # atom z-coordinates
        CIF.A_occupancy.value       # occupancy
    ]]

    # SORT ROWS BY SEQUENCE NUMBERING BY RESIDUE (SEQ ID) THEN BY ATOMS (A_ID) AND THEN CHAIN:
    pdf_merged.reset_index(drop=True, inplace=True)
    pdf_merged = pdf_merged.sort_values([CIF.A_label_asym_id.value, CIF.S_seq_id.value, CIF.A_id.value])

    # ONLY KEEP THESE COLUMNS AND IN THIS ORDERING:
    pdf_merged = pdf_merged[[CIF.A_label_asym_id.value,
                             CIF.S_seq_id.value,
                             CIF.A_id.value,
                             CIF.S_mon_id.value,
                             CIF.A_label_atom_id.value,
                             CIF.A_Cartn_x.value,
                             CIF.A_Cartn_y.value,
                             CIF.A_Cartn_z.value]]
    return pdf_merged
