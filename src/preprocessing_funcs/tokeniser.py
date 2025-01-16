"""
TOKENISER.PY
    - CALL `cif_parser.py` TO READ IN AND PARSE mmCIF FILE(S) TO EXTRACT THE 14 FIELDS.
    - WRITE TO .SSV FLATFILE.
    - ENUMERATE ATOMS AND AMINO ACID RESIDUES.
    - SUBTRACT COORDINATES BY THEIR MEAN COORDINATE VALUES PER ATOM.
----------------------------------------------------------------------------------------------------------------------
The following 14 mmCIF fields are extracted from the raw mmCIF files, parsed and tokenised into a dataframe.
The 14 fields are:

_atom_site:
    group_PDB           # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
    label_seq_id        # RESIDUE POSITION      - JOIN TO S_seq_id, THEN REMOVE IT.
    label_comp_id       # RESIDUE (3-LETTER)    - FOR SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
    id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DATAFRAME.
    label_atom_id       # ATOM                  - KEEP IN DATAFRAME.
    label_asym_id       # CHAIN                 - JOIN TO S_asym_id, KEEP IN DATAFRAME.
    Cartn_x             # COORDINATES           - ATOM X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_y             # COORDINATES           - ATOM Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_z             # COORDINATES           - ATOM Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    occupancy           # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.

_pdbx_poly_seq_scheme:
    seq_id              # RESIDUE POSITION      - JOIN TO A_label_seq_id. SORT ON THIS, KEEP IN DATAFRAME.
    mon_id              # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DATAFRAME.
    pdb_seq_num         # RESIDUE POSITION      - KEEP FOR NOW, AS MAY RELATE TO INPUT TO MAKE EMBEDDINGS.
    asym_id             # CHAIN                 - JOIN TO A_label_asym_id, SORT ON THIS, THEN REMOVE IT.

----------------------------------------------------------------------------------------------------------------------
The output of the current `parse_tokenise_cif_write_flatfile()` function is a 17-column dataframe.
'_atom_site' is abbreviated to 'A_' prefix.
'_pdbx_poly_seq_scheme' is abbreviated to 'S_' prefix.
These 17 columns are:

A_label_asym_id       # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
S_seq_id              # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
A_id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
A_label_atom_id       # ATOM                  - KEEP IN DF.
A_Cartn_x             # COORDINATES           - ATOM X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_y             # COORDINATES           - ATOM Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_z             # COORDINATES           - ATOM Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
aa_label_num          # ENUMERATED RESIDUES   - EQUIVALENT TO `ntcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
bb_or_sc              # BACKBONE OR SIDE-CHAIN ATOM ('bb' or 'sc'), KEEP FOR POSSIBLE SUBSEQUENT OPERATIONS.
bb_atom_pos           # ATOM POSITION CA OR MOST C-TERM OTHER BB ATOM, PER RESIDUE. KEEP IN DF.
bbindices             # INDEX POSITION OF THE ATOM POSITION (`A_id`) OF ALLOCATED BACKBONE ATOM.
atom_label_num        # ENUMERATED ATOMS      - EQUIVALENT TO `atomcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
aa_atom_tuple         # RESIDUE-ATOM PAIR     - ONE TUPLE PER ROW. KEEP IN DF.
aa_atom_label_num     # ENUMERATED RESIDUE-ATOM PAIRS. (ALTERNATIVE WAY TO GENERATE `atomcodes`, will be `aaatomcodes`).
mean_xyz              # MEAN OF COORDINATES   - MEAN OF X, Y, Z COORDINATES FOR EACH ATOM. KEEP IN DF TEMPORARILY.
mean_corrected_x      # X COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_y      # Y COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_z      # Z COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.

"""


import os
import re
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from math import sqrt
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api

# If more than this proportion of residues have no backbone atoms, remove the chain.
MIN_RATIO_MISSING_BACKBONE_ATOMS = 0.0


def _torch_load_embed_pt(pt_fname: str):
    """
    Load torch embedding `.pt` file.
    :param pt_fname: Filename of `.pt` file to be loaded.
    :return: A Tensor or a dict.
    """
    abs_path = os.path.dirname(os.path.abspath(__file__))
    pt_fname = pt_fname.removesuffix('.pt')
    path_pt_fname = f'../diffusion/diff_data/emb/{pt_fname}.pt'
    abspath_pt_fname = os.path.normpath(os.path.join(abs_path, path_pt_fname))
    assert os.path.exists(abspath_pt_fname), f"'{pt_fname}.pt' does not exist at '{abspath_pt_fname}'"
    pt = None
    try:
        pt = torch.load(abspath_pt_fname, weights_only=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: '{abspath_pt_fname}'.") from e
    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(f"Failed to unpickle file: '{abspath_pt_fname}'. "
                         f"May be corrupted or not valid PyTorch file.") from e
    except EOFError as e:
        raise EOFError(f"Unexpected end of file while loading file: '{abspath_pt_fname}'. "
                       f"It might be incomplete or corrupted.") from e
    except RuntimeError as e:
        raise RuntimeError(f"PyTorch encountered an issue while loading file: '{abspath_pt_fname}'. "
                           f"Details: {str(e)}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading file: '{abspath_pt_fname}'. "
                        f"Error: {str(e)}") from e

    assert pt is not None, f"'{pt_fname}' seems to have loaded successfully but is null..."

    if isinstance(pt, dict):
        assert bool(pt), f"The dict of '{pt_fname}' was read in successfully but seems to be empty..."
    else:
        if isinstance(pt, torch.Tensor):
            assert pt.numel() > 0, f"The tensor of '{pt_fname}' was read in successfully but seems to be empty..."
    return pt


def get_nums_of_missing_data(pdf):
    pd_isna = int((pdf.map(lambda x: isinstance(x, float) and pd.isna(x))).sum().sum())
    pd_na = int((pdf.map(lambda x: x is pd.NA)).sum().sum())
    pd_nat = int((pdf.map(lambda x: x is pd.NaT)).sum().sum())
    np_nan = int((pdf.map(lambda x: x is np.nan)).sum().sum())
    none = int((pdf.map(lambda x: x is None)).sum().sum())
    empty_str = int((pdf.map(lambda x: x == ' ')).sum().sum())
    na_str = int((pdf.map(lambda x: x == 'na' or x == 'NA')).sum().sum())
    nan_str = int((pdf.map(lambda x: x == 'nan' or x == 'NAN' or x == 'NaN')).sum().sum())
    none_str = int((pdf.map(lambda x: x == 'none' or x == 'None')).sum().sum())
    null_str = int((pdf.map(lambda x: x == 'null' or x == 'Null')).sum().sum())

    counts = {
        'NaN': pd_isna,
        'pd.NA': pd_na,
        'pd.NaT': pd_nat,
        'np.nan': np_nan,
        'None': none,
        ' ': empty_str,
        "na": na_str,
        "nan": nan_str,
        "none": none_str,
        "null": null_str
    }
    print(counts)
    has_missing_data = any(value > 0 for value in counts.values())

    if has_missing_data:
        nan_positions = pdf.map(lambda x: (isinstance(x, float) and pd.isna(x) or
                                           x is pd.NA or
                                           x is pd.NaT or
                                           x is np.nan or
                                           x is None or
                                           x == ' ' or
                                           x in ['na', 'NA'] or
                                           x in ['nan', 'NAN', 'NaN'] or
                                           x in ['none', 'None'] or
                                           x in ['null', 'Null']
                                           ))

        # Identify the row (index) and column names where the above checks are True
        nan_positions = nan_positions.stack().reset_index()  # Convert to long format
        nan_positions = nan_positions[nan_positions[0]]  # Filter only rows where True
        nan_positions = nan_positions[['level_0', 'level_1']]  # Keep only index and column info
        nan_positions.columns = ['index', 'column']  # Rename columns
        raise ValueError('There are missing values.. needs to be addressed.')

    # missing_strings = ['NaN', 'None', 'N/A', 'missing', 'NULL', '']
    # pdf = pdf.replace(missing_strings, np.nan)
    return pdf


def _each_column_has_expected_values(pdf_chain):
    pdf_chain.head()
    # TODO checks that each column has values of the expected type and range.
    pass


def _assign_mean_corrected_coordinates(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: calc mean-corrected coords')
    result_pdfs = list()
    for pdf in pdfs:
        # SUBTRACT EACH COORDINATE BY THE MEAN OF ALL 3 PER ATOM:
        pdf.loc[:, 'mean_xyz'] = pdf[['A_Cartn_x', 'A_Cartn_y', 'A_Cartn_z']].mean(axis=1)
        pdf.loc[:, 'mean_corrected_x'] = pdf['A_Cartn_x'] - pdf['mean_xyz']
        pdf.loc[:, 'mean_corrected_y'] = pdf['A_Cartn_y'] - pdf['mean_xyz']
        pdf.loc[:, 'mean_corrected_z'] = pdf['A_Cartn_z'] - pdf['mean_xyz']
        expected_num_of_cols = 18
        assert len(pdf.columns) == expected_num_of_cols, \
            f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
        result_pdfs.append(pdf)
    return result_pdfs


def _enumerate_residues_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate residue-atom pairs of CIF for one protein, by mapping via pre-written json at `data/enumeration`. Add
    this enumeration to a new column `aa_atom_label_num`. It serves as the tokenised form of polypeptide residue-atom
    pairs for this protein, to be read later to `aaatomcodes` array.
    :param pdf: Dataframe of one protein CIF, containing atoms to enumerate to new column.
    :return: Given dataframe with two new columns holding the enumerated residue-atom pairs data, as well as a column
    holding the intermediate data of residue-atom pairs. Expected to have 14 columns.
    """
    residues_atoms_enumerated = dh.read_enumerations_json(fname='residues_atoms_no_hydrogens')
    # CAST THE STRING REPRESENTATION OF A TUPLE TO AN ACTUAL TUPLE FOR KEY TO WORK IN MAPPING:
    residues_atoms_enumerated = {eval(k): v for k, v in residues_atoms_enumerated.items()}
    # FIRST MAKE NEW COLUMN OF RESIDUE-ATOM PAIRS. E.G. CONTAINS ('ASP':'C'), ('ASP':'CA'), ETC:
    pdf['aa_atom_tuple'] = list(zip(pdf['S_mon_id'], pdf['A_label_atom_id']))
    # MAKE NEW COLUMN FOR ENUMERATED RESIDUE-ATOM PAIRS, VIA RESIDUE-ATOM PAIRS:
    pdf['aa_atom_label_num'] = (pdf['aa_atom_tuple']
                                .map(residues_atoms_enumerated)
                                .astype('Int64'))
    expected_num_of_cols = 14
    assert len(pdf.columns) == expected_num_of_cols, \
        f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
    return pdf


def _enumerate_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate atoms of mmCIF for one protein, by mapping via pre-written json at `data/enumeration`. Add
    this enumeration to a new column `atom_label_num`. It serves as the tokenised form of polypeptide atoms for this
    protein, to be read later to `atomcodes` array.
    :param pdf: Dataframe of one protein CIF, containing atoms to enumerate to new column.
    :return: Given dataframe with one new column holding the enumerated atoms data. Expected to have 12 columns.
    """
    # MAKE NEW COLUMN FOR ENUMERATED ATOMS ('C', 'CA', ETC), USING JSON->DICT, CAST TO INT:
    atoms_enumerated = dh.read_enumerations_json(fname='unique_atoms_only_no_hydrogens')
    pdf['atom_label_num'] = (pdf['A_label_atom_id']
                             .map(atoms_enumerated)
                             .astype('Int64'))
    expected_num_of_cols = 12
    assert len(pdf.columns) == expected_num_of_cols, \
        f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
    return pdf


def _enumerate_residues(pdf: pd.DataFrame) -> pd.DataFrame:
    # MAKE NEW COLUMN FOR ENUMERATED RESIDUES, USING JSON->DICT, CAST TO INT.
    # `residues_enumerated` DICT KEY AND `S_mon_id` COLUMN VALUES MAP VIA 3-LETTER RESIDUE NAMES:
    residues_enumerated = dh.read_enumerations_json(fname='residues')
    pdf.loc[:, 'aa_label_num'] = (pdf['S_mon_id']
                                  .map(residues_enumerated)
                                  .astype('Int64'))
    expected_num_of_cols = 11
    assert len(pdf.columns) == expected_num_of_cols, \
        f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
    return pdf


def _enumerate_atoms_and_residues(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    """
    Enumerate residues, atoms, and residue-atoms pairs of given protein CIF data, and store to new columns in given
    dataframe. Currently hard-coded to only use atom data that lacks all hydrogen atoms.
    :param pdfs: List of dataframes of one protein CIF, per chain, containing atoms to enumerate to new columns.
    :return: Dataframe with new columns of enumerated data for residues, atoms, and residue-atoms pairs.
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: enumerate atoms and residues')
    result_pdfs = list()
    for pdf in pdfs:
        pdf = _enumerate_residues(pdf)
        pdf = _enumerate_atoms(pdf)
        pdf = _enumerate_residues_atoms(pdf)
        result_pdfs.append(pdf)
    return result_pdfs


def _assign_backbone_index_to_all_residue_rows(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: assign bb indices')

    result_pdfs = list()

    for pdf in pdfs:

        # ASSIGN INDEX OF CHOSEN BACKBONE ATOM (ALPHA-CARBON) FOR ALL ROWS IN EACH ROW-WISE-RESIDUE SUBSETS:
        chain = pdf['S_asym_id'].unique()[0]

        # Even though `a_id` is always an int (when no NaNs), and I cast to Int64 below. The column remains float64.
        # Apparently I have to cast this column beforehand ? Very odd behaviour by Pandas.
        pdf['bb_atom_pos'] = pd.Series(dtype='Int64')

        for S_seq_id, aa_group in pdf.groupby('S_seq_id'):  # GROUP BY RESIDUE POSITION VALUE
            # GET ATOM INDEX ('A_id') WHERE ATOM ('A_label_atom_id') IS 'CA' IN THIS RESIDUE GROUP.
            a_id_of_CA = aa_group.loc[aa_group['A_label_atom_id'] == 'CA', 'A_id']

            # IF NO 'CA' FOR THIS RESIDUE, USE MOST C-TERMINAL NON-CA BACKBONE ATOM POSITION INSTEAD:
            if a_id_of_CA.empty:
                print(f"No 'CA' for {aa_group['S_mon_id'].iloc[0]} at {S_seq_id} "
                      f"(PDBid={pdb_id}, chain={chain})")
                positions_of_all_bb_atoms = aa_group.loc[aa_group['bb_or_sc'] == 'bb', 'A_id'].to_numpy()

                # IF NO BACKBONE ATOMS FOR THIS RESIDUE AT ALL, REMOVE THIS RESIDUE FROM THIS CIF:
                if positions_of_all_bb_atoms.size == 0:
                    aa = {aa_group['S_mon_id'].iloc[0]}
                    print(f'{aa} at {S_seq_id} only has atoms {str(list(aa_group['A_label_atom_id']))}. '
                          f'Hence, no backbone atoms at all, so {aa} at {S_seq_id} will be completely removed from '
                          f'this dataframe.')
                    pdf = pdf[pdf['S_seq_id'] != S_seq_id]
                    continue  # to next residue
                else:
                    a_id = max(positions_of_all_bb_atoms)
                    print(f'Instead, assigning position of most C-terminal non-CA backbone atom={a_id}.')
                    most_cterm_bb_atom = aa_group.loc[aa_group['A_id'] == a_id, 'A_label_atom_id'].values[0]
                    print(f'Non-CA backbone atoms for this residue are at: {str(list(positions_of_all_bb_atoms))}, '
                          f'so {a_id} is selected. (The atom at this position is: {most_cterm_bb_atom}.)')
                    pdf.loc[aa_group.index, 'bb_atom_pos'] = a_id
                    continue
            else:
                a_id = a_id_of_CA.iloc[0]

                # ASSIGN THIS ATOM INDEX TO BB_ATOM_POS ('bb_atom_pos') FOR ALL ROWS IN THIS GROUP:
            pdf.loc[aa_group.index, 'bb_atom_pos'] = a_id

        # CAST NEW COLUMN TO INT64 (FOR CONSISTENCY):
        print(f'Type of `bb_atom_pos` column, before any casting operations={pdf['bb_atom_pos'].dtype}')

        pdf.loc[:, 'bb_atom_pos'] = pd.to_numeric(pdf['bb_atom_pos'], errors='coerce')
        print(f'`b_atom_pos` column should be numeric type={pdf['bb_atom_pos'].dtype}')

        pdf.loc[:, 'bb_atom_pos'] = pdf['bb_atom_pos'].astype('Int64')
        print(f'`bb_atom_pos` column should be integer type={pdf['bb_atom_pos'].dtype}')
        result_pdfs.append(pdf)

    return result_pdfs


def _select_chains_to_use(pdfs: List[pd.DataFrame], chains: list = None, pdb_id: str = None) -> List[pd.DataFrame]:
    """
    Select which chains to keep for further parsing and tokenisation and to be written to flatfile. If no chains
    specified, all protein chains will be kept. (If no PDBid is given, just don't print anything).
    NOTE: Currently this function is only performing a simple chain selection, just the first in the given list.
    :param pdfs: Dataframes, one per chain, for given protein CIF.
    :param chains: One of more chain(s) to keep. e.g. [A, C]. <Currently not used>
    :param pdb_id: Only for printing out as part of tracking function calls.
    :return: A list of one or more dataframes, according to which chains to keep (This is also temporary to avoid
    breaking subsequent operations).
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: select which chains to keep.')
    if len(pdfs) > 0:
        pdfs = [pdfs[0]]
    return pdfs


def _only_keep_chains_with_enough_bb_atoms(pdfs: List[pd.DataFrame], pdb_id: str = None) -> List[pd.DataFrame]:
    """
    Remove chain(s) from list of chains for this CIF if not more than a specific number of backbone polypeptides.
    :param pdfs: List of dataframes, one per chain, for one CIF.
    :param pdb_id: The PDB id of the corresponding CIF data.
    :return: List of dataframes, one per chain of protein, with any chains removed if they have less than the minimum
    permitted ratio of missing atoms (arbitrarily chosen for now).
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: remove chains without enough backbone atoms')
    result_pdfs = []
    for pdf in pdfs:
        aa_w_no_bbatom_count = (pdf
                                .groupby('S_seq_id')['bb_or_sc']
                                .apply(lambda x: 'bb' not in x.values)
                                .sum())
        total_atom_count = pdf.shape[0]
        if (aa_w_no_bbatom_count / total_atom_count) <= MIN_RATIO_MISSING_BACKBONE_ATOMS:
            result_pdfs.append(pdf)
    if len(result_pdfs) == 0:
        print(f'PDBid={pdb_id}: After removing chains that have too many residues that lack any backbone atom '
              f'coordinates at all, there are no chains left for this protein, PDBid={pdb_id}. It needs to be removed '
              f'from the dataset entirely.')
    return result_pdfs


def _only_keep_chains_of_polypeptide(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    """
    Remove chain(s) from list of chains for this CIF if not polypeptide. Takes advantage of output of previous function
    which assigns to a new column called `bb_or_sc` 'bb' for polypeptide backbone atoms, 'sc' for polypeptide
    side-chain atoms, otherwise `pd.NaT` which therefore indicates the chain is not polypeptide.
    (RCSB CIF assigns a different chain to different molecule in a complex, e.g. RNA-protein complex, so if one row in
    the aforementioned column is `pd.NaT`, you should find that all are `pd.NaT`).
    :param pdfs: List of dataframes, one per chain, for one CIF.
    :param pdb_id: The PDB id of the corresponding CIF data.
    :return: List of dataframes, one per chain of protein, with any non-polypeptide chains removed.
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: remove non-protein chains')
    result_pdfs = []
    for pdf in pdfs:
        try:
            chain = pdf['S_asym_id'].iloc[0]
            atleast_one_row_isna = pdf['bb_or_sc' ].isna().any()
            all_rows_isna = pdf['bb_or_sc' ].isna().all()
            if atleast_one_row_isna:
                nat_indices = pdf[pd.isna(pdf['bb_or_sc' ])].index
                print(f'nat_indices={nat_indices}')
                print(f'There are atoms in chain={chain} of PDB id={pdb_id} which are not polypeptide atoms, so this chain '
                      f'will be excluded.')
                if not all_rows_isna:
                    print(f"It seems that while at least one row in column 'bb_or_sc' has null, "
                          f'not all rows are null. This is unexpected and should be investigated further. '
                          f'(Chain {chain} of PDB id {pdb_id}).')
            else:
                result_pdfs.append(pdf)
            if len(result_pdfs) == 0:
                print(f'PDBid={pdb_id}: After removing all non-polypeptide chains, there are no chains left. '
                      f'This should not occur, so needs to be investigated further.')
        except IndexError:
            print(f" `chain = pdf['S_asym_id'].iloc[0]` fails. "
                  f"\npdf['S_asym_id']={pdf['S_asym_id']}")
    return result_pdfs


def _make_new_bckbone_or_sdchain_col(pdfs: List[pd.DataFrame], pdb_id: str=None) -> List[pd.DataFrame]:
    BACKBONE = ('N', 'CA', 'C', 'O', 'OXT')  # AKA "MAIN-CHAIN"
    SIDECHAIN = ('CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2',
                 'CZ3', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2', 'OG',
                 'OG1', 'OG2', 'OH', 'SD', 'SG')

    if pdb_id:
        print(f'PDBid={pdb_id}: make new column `bb_or_sc` - indicates whether atom is backbone or side-chain.')

    result_pdfs = list()

    for pdf in pdfs:
        is_backbone_atom = pdf['A_label_atom_id'].isin(BACKBONE)
        is_sidechain_atom = pdf['A_label_atom_id'].isin(SIDECHAIN)

        # MAKE NEW COLUMN TO INDICATE IF ATOM IS FROM POLYPEPTIDE BACKBONE ('bb) OR SIDE-CHAIN ('sc'):
        pdf.loc[:, 'bb_or_sc'] = np.select([is_backbone_atom, is_sidechain_atom], ['bb', 'sc'], default='placeholder')
        pdf.loc[pdf['bb_or_sc'] == 'placeholder', 'bb_or_sc'] = pd.NA

        expected_num_of_cols = 9
        assert len(pdf.columns) == expected_num_of_cols, \
            f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
        result_pdfs.append(pdf)

    return result_pdfs


def _remove_all_hydrogen_atoms(pdfs: List[pd.DataFrame], pdb_id: str=None) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: remove Hydrogens')
    hydrogen_atoms = dh.read_lst_file_from_data_dir('enumeration/hydrogens.lst')
    result_pdfs = []
    for pdf in pdfs:
        _pdf = pdf.loc[~pdf['A_label_atom_id'].isin(hydrogen_atoms)]
        result_pdfs.append(_pdf)
    return result_pdfs


def _keep_only_the_given_chain(pdfs: List[pd.DataFrame], chain: str, pdb_id: str=None) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: only keep the specified chain={chain}.')
    result_pdfs = list()
    for pdf in pdfs:
        if pdf['S_asym_id'].isin([chain]).any():
            result_pdfs.append(pdf)
            return result_pdfs
    return result_pdfs


def __fetch_mmcif_from_pdb_api_and_write(pdb_id: str, relpath_dst_cif: str) -> None:
    """
    Fetch raw mmCIF data from API (expected hosted at 'https://files.rcsb.org/download/') using given PDB id,
    and write out to flat file.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param relpath_dst_cif: Relative path, filename and `.cif` extension of CIF to write.
    """
    response = api.call_for_cif_with_pdb_id(pdb_id)
    with open(relpath_dst_cif, 'w') as file:
        file.write(response.text)


def _get_mmcif_data(pdb_id: str, relpath_cif_dir: str) -> dict:
    """
    Read in mmCIF file of specified PDB id, to Python dict. mmCIF file of specified PDB id will be read
    from the specified relative path, if it has previously been downloaded to there, otherwise it will be downloaded
    directly from the rcsb server.
    :param pdb_id: PDB id of mmCIF file to read in.
    :param relpath_cif_dir: Relative path to mmCIF files directory.
    :return: Python dict of the specified mmCIF file.
    """
    relpath_cif_dir = relpath_cif_dir.removesuffix('.cif').removeprefix('/').removesuffix('/')
    pdb_id = pdb_id.removesuffix('.cif')
    relpath_cif = f'{relpath_cif_dir}/{pdb_id}.cif'
    if os.path.exists(relpath_cif):
        try:
            MMCIF2Dict(relpath_cif)
        except ValueError:
            print(f"{pdb_id}.cif appears to be empty. "
                  f"Attempt to read {pdb_id} directly from 'https://files.rcsb.org/download/'{pdb_id}")
            __fetch_mmcif_from_pdb_api_and_write(pdb_id=pdb_id, relpath_dst_cif=relpath_cif)
    else:
        print(f'Did not find this CIF locally ({relpath_cif}). '
              f"Attempt to read {pdb_id} directly from 'https://files.rcsb.org/download/'{pdb_id}")
        __fetch_mmcif_from_pdb_api_and_write(pdb_id=pdb_id, relpath_dst_cif=relpath_cif)
    mmcif = MMCIF2Dict(relpath_cif)
    if not mmcif:
        print(f'{relpath_cif}/{pdb_id}.cif appears to be empty. ')
    return mmcif


def __split_pdbid_chain(pdbid_chain):
    match = re.match(r"^(.*)_([A-Za-z])$", pdbid_chain)
    if match:
        pdbid, chain = match.groups()
        return pdbid, chain
    else:
        return pdbid_chain, None


def _generate_list_of_pdbids_in_cif_dir(path_cif_dir: str) -> list:
    cifs = glob.glob(os.path.join(path_cif_dir, f'*.cif'))
    path_cifs = [cif.upper() for cif in cifs if os.path.isfile(cif)]
    pdb_id_list = []

    for path_cif in path_cifs:
        cif_basename = os.path.basename(path_cif)
        pdbid = os.path.splitext(cif_basename)[0]
        pdb_id_list.append(pdbid)
    return pdb_id_list


def parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir='../diffusion/diff_data/mmCIF',
                                          relpath_toknsd_ssv_dir='../diffusion/diff_data/tokenised',
                                          relpath_pdblst: str = None,
                                          flatfile_format_to_write: str = 'ssv',
                                          pdb_ids=None,
                                          write_lst_file=True) -> Tuple[List[pd.DataFrame], str]:
    """
    Parse and tokenise sequence and structure fields from mmCIFs of proteins, returning a list of Pandas dataframes,
    with one dataframe per chain. Write each dataframe to one flat file (ssv by default), in `tokenised` directory,
    The mmCIFs to parse, tokenise and write to flat flats is specified by one or more of the following 3:
      - Relative path of directory containing pre-downloaded CIF files, e.g. `diff_data/mmCIF`, reading all CIFs in.
      - Relative path of a list file of PDB ids e.g. `diff_data/SD_573.lst`.
      - One PDB id or a Python list of PDB ids.
    "Parsing" involves extracting required fields from mmCIF files to dataframes.
    "Tokenising" involves enumerating atoms and residues, and calculating mean-adjusted x, y, z coordinates.
    Where a CIF has > 1 chain, parse & tokenise all polypeptide-only chains & only those with sufficient backbone
    coordinates, according to some pre-decided threshold constant (i.e. `MIN_RATIO_MISSING_BACKBONE_ATOMS`).
    :param relpath_cif_dir: Relative path of directory containing pre-downloaded raw mmCIF files.
    :param relpath_toknsd_ssv_dir: Relative path of directory for tokenised CIFs as flat files.
    :param relpath_pdblst: <OPTIONAL> Relative path to list file of one or more PDB ids or PDBid_chains names. e.g.
    `globins_10.lst`.
    :param flatfile_format_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param pdb_ids: <OPTIONAL> PDB id(s) to parse & tokenised. Expect either string of PDB id or list of PDB ids.
    Otherwise, just read from `src/diffusion/diff_data/mmCIF` subdir by default.
    Use `src/diffusion/diff_data/tokenised` by default, (hence expecting cwd to be `src/diffusion`).
    :param write_lst_file: True to write out a new PDBids_chain `.lst` file of all those that have been parsed and
    tokenised. It is subsequently passed (via its path & file name) to `main()` function in
    `pytorch_protfold_allatomclustrain_singlegpu.py` script.
    :return: List of dataframes, one per chain. Each dataframe has the parsed and tokenised data of one CIF file
     and each is also written to a flatfile (ssv by default) at `src/diffusion/diff_data/tokenised`.
    Dataframe currently has these 17 Columns: ['A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x',
    'A_Cartn_y', 'A_Cartn_z', 'aa_label_num', 'bb_or_sc', 'bb_atom_pos', 'atom_label_num', 'aa_atom_tuple',
    'aa_atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z'].
    """
    # INIT LIST OF PARSED PDB DATA IN DATAFRAMES (ONE DATAFRAME PER PDB AND PER CHAIN):
    cif_pdfs_per_chain = []  # THIS IS RETURNED.

    # THIS IS JUST A LIST OF PDBID_CHAIN (MADE UP OF COMBO OF ANY IN TOKENISED AND NEWLY PARSED):
    for_pdbchain_lst = []

    # BUILD LIST OF PDBIDS TO PROCESS, FROM EITHER GIVEN STRING, LIST, .LST FILE OR PATH TO CIFS:
    if pdb_ids:
        if isinstance(pdb_ids, str):  # If string, it is assumed that this is one PDB id.
            pdb_ids = [pdb_ids]
    if relpath_pdblst:
        if pdb_ids is None:
            pdb_ids = []
        pdb_ids.extend(dh.read_pdb_lst_file(relpath_pdblst=relpath_pdblst))
    if relpath_cif_dir:
        if pdb_ids is None:
            pdb_ids = []
        relpath_cif_dir = relpath_cif_dir.removesuffix('/').removeprefix('/')
        assert os.path.exists(relpath_cif_dir), 'Not found `relpath_cif_dir`.'
        _pdb_ids = _generate_list_of_pdbids_in_cif_dir(path_cif_dir=relpath_cif_dir)
        pdb_ids.extend(_pdb_ids)

    pdb_ids = list(set(pdb_ids))
    assert pdb_ids is not None, 'This is a bug! `pdb_ids` is None ?! It must be a list of at least one PDB id.'
    assert len(pdb_ids) > 0, 'This is a bug! `pdb_ids` is an empty list ?! It must be a list of at least one PDB id.'

    # MAKE A LIST OF PDBIDS THAT ARE SSVS IN TOKENISED DIR (I.E. HAVE ALREADY BEEN TOKENISED):
    cif_tokenised_ssv_dir = '../diffusion/diff_data/tokenised'
    pdbid_chains_of_pretokenised = []
    if os.path.exists(cif_tokenised_ssv_dir):
        pdbid_chains_of_pretokenised = [item.removesuffix('.ssv')
                                        for item in os.listdir(cif_tokenised_ssv_dir)
                                        if item.endswith('.ssv')]

    for pdb_id in pdb_ids:  # Expecting only one PDB id per line. (Can be PDBid_chain or just PDBid.)
        pdb_id = pdb_id.rstrip().split()[0]
        flatfile_format_to_write = flatfile_format_to_write.removeprefix('.').lower()
        pdb_id = pdb_id.removesuffix('.cif')
        print(f'Starting PDBid={pdb_id} ---------------------------------------------------------')
        if pdb_id == '5TJ5':
            print(f'{pdb_id} is known to have 2500 missing entries in the aa sequence field! So {pdb_id} will be '
                  f'excluded.')
            continue

        # THE LOGIC EXPECTS THAT IF IT HAS BEEN TOKENISED AND SAVED AS ssv, THEN IT WILL BE BY THE CHAIN AND SO
        # MUST HAVE THE SUFFIX `<pdbid>_A.ssv`, `<pdbid>_B.ssv`, ETC, RATHER THAN JUST `<pdbid>.ssv`:
        if pdb_id in pdbid_chains_of_pretokenised:
            pdbid_chain = pdb_id  # to make it clear that this is expected to include a chain suffix.
            print(f'PDBid={pdbid_chain} already tokenised. Reading ssv in to pdf. Add to list.')
            _cif_pdfs_per_chain = dh.read_tokenised_cif_ssv_to_pdf(relpath_tokensd_dir=relpath_toknsd_ssv_dir,
                                                                   pdb_id=pdbid_chain)
            # FURTHERMORE IT IS EXPECTED THERE IS *ONLY ONE CHAIN* PER PDB ID, SO IT IS A PYTHON LIST OF ONE PDB_ID.
            assert _cif_pdfs_per_chain, 'no cif_pdfs_per_chain returned. Expected list of one pdf for one chain.'
            for_pdbchain_lst.append(pdbid_chain)
            continue
        # OTHERWISE GET THE CIF (LOCALLY OR VIA API), AND EXPECT PDBid TO BE WITHOUT THE CHAIN:

        pdbid, chain = __split_pdbid_chain(pdb_id)  # Note: chain should be empty string
        mmcif_dict = _get_mmcif_data(relpath_cif_dir=relpath_cif_dir, pdb_id=pdbid)

        # ** AND FINALLY, YOU CAN START PARSING THE CIFS (THAT HAVE NOT ALREADY BEEN PARSED) **
        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT & JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        # (THIS RETURNS A LIST OF DATAFRAMES, ONE PER POLYPEPTIDE CHAIN).
        cif_pdfs_per_chain = parser.parse_cif(mmcif_dict=mmcif_dict, pdb_id=pdbid)
        if chain:
            cif_pdfs_per_chain = _keep_only_the_given_chain(pdfs=cif_pdfs_per_chain, chain=chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _remove_all_hydrogen_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _make_new_bckbone_or_sdchain_col(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_of_polypeptide(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_with_enough_bb_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _select_chains_to_use(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)

        if len(cif_pdfs_per_chain) == 0:
            print(f'No chains left for this CIF: PDBid={pdbid}, so it cannot be used.')
            continue

        cif_pdfs_per_chain = _assign_backbone_index_to_all_residue_rows(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _enumerate_atoms_and_residues(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _assign_mean_corrected_coordinates(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        dh.write_tokenised_cifs_to_flatfiles(pdfs=cif_pdfs_per_chain,
                                             dst_data_dir=relpath_toknsd_ssv_dir,
                                             flatfiles=flatfile_format_to_write, pdb_id=pdbid)
        # ADD CHAIN TO NAME OF PDB ID:
        for pdf_chain in cif_pdfs_per_chain:
            get_nums_of_missing_data(pdf_chain)
            _each_column_has_expected_values(pdf_chain)
            chain = pdf_chain['S_asym_id'].iloc[0]
            pdbid_chain = f'{pdbid}_{chain}'
            for_pdbchain_lst.append(pdbid_chain)

    lst_file = f'pdbchains_{len(for_pdbchain_lst)}.lst'
    if write_lst_file:
        dh.write_list_to_lst_file(list_to_write=for_pdbchain_lst,
                                  path_fname=f'../diffusion/diff_data/PDBid_list/{lst_file}')
    return cif_pdfs_per_chain, lst_file


def _chdir_to_tokeniser() -> str:
    """
    Using relative paths and the possibility of calling this script from two different locations necessitates
    temporarily changing current working directory before changing back at the end.
    :return: Current working directory before changing to this one with absolute path.
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f'Path changed from {cwd} to = {os.getcwd()}. (This is intended to be temporary).')
    return cwd


def load_dataset(targetfile_lst_path: str) -> Tuple[List, List]:
    assert os.path.exists(targetfile_lst_path), f'{targetfile_lst_path} cannot be found. (Btw, your cwd={os.getcwd()})'

    print('Starting `load_dataset()`...')
    cwd = _chdir_to_tokeniser()  # Change current dir. (Store cwd to return to at end).
    tnum = 0
    sum_d2 = 0
    sum_d = 0
    nn = 0

    # GET THE LIST OF PDB NAMES FOR PROTEINS TO TOKENISE:
    targetfile = ''
    try:
        with open(targetfile_lst_path, 'r') as lst_f:
            targetfile = [line.strip() for line in lst_f]
    except FileNotFoundError:
        print(f'{lst_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")

    train_list, validation_list = [], []
    abnormal_structures = []  # list any structures with invalid interatomic distances of consecutive residues

    for line in targetfile:  # Expecting only one PDB id per line.
        sp = []
        target_pdbid = line.rstrip().split()[0]
        print(f'Read in {target_pdbid}.ssv')
        pdf_target = pd.read_csv(f'{'../diffusion/diff_data/tokenised'}/{target_pdbid}.ssv', sep=' ')

        # GET MEAN-CORRECTED COORDINATES VIA 'mean_corrected_x', '_y', '_z' TO 3-ELEMENT LIST:
        coords = pdf_target[['mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z']].values
        len_coords = len(coords)  # should be same as..? TODO

        # GET `atomcodes` VIA 'atom_label_num' COLUMN, WHICH HOLDS ENUMERATED ATOMS VALUES:
        atomcodes = pdf_target['atom_label_num'].tolist()

        # THE FOLLOWING LINE INCREMENTS BY 1 FOR EVERY SUBSEQUENT RESIDUE, SUCH THAT FOR A PDB OF 100 RESIDUES,
        # REGARDLESS OF WHAT RESIDUE THE STRUCTURAL DATA STARTS FROM AND REGARDLESS OF ANY GAPS IN THE SEQUENCE,
        # `aaindices` STARTS FROM 0, INCREMENTS BY 1 AND ENDS AT 99:
        pdf_target['aaindices'] = pd.factorize(pdf_target['S_seq_id'])[0]
        aaindices = pdf_target['aaindices'].tolist()

        # ASSIGN DATAFRAME INDEX OF BACKBONE ATOM POSITION PER RESIDUE IN NEW COLUMN `BBINDICES` FOR REF TO ATOMS:
        atom_positions_rowindex_map = {value: index for index, value in pdf_target['A_id'].items()}
        pdf_target['bbindices'] = pdf_target['bb_atom_pos'].map(atom_positions_rowindex_map)

        # DE-DUPLICATE ROWS ON RESIDUE POSITION (`S_seq_id`) TO GET CORRECT DIMENSION OF `aacodes` and `bbindices`:
        pdf_target_deduped = (pdf_target
                              .drop_duplicates(subset='S_seq_id', keep='first')
                              .reset_index(drop=True))

        # GET `aacodes`, VIA 'aa_label_num' COLUMN, WHICH HOLDS ENUMERATED RESIDUES VALUES:
        aacodes = pdf_target_deduped['aa_label_num'].tolist()

        bbindices = pdf_target_deduped['bbindices'].tolist()

        # ONLY INCLUDE PROTEINS WITHIN A CERTAIN SIZE RANGE:
        if len(aacodes) < 10 or len(aacodes) > 500:
            print(f'{target_pdbid} CIF is {len(aacodes)} residues long. It is not within the chosen range 10-500 '
                  f'residues, so will be excluded.')
            continue

        # READ PRE-COMPUTED EMBEDDING OF THIS PROTEIN:
        pdb_embed = _torch_load_embed_pt(pt_fname=target_pdbid)

        # AND MAKE SURE IT HAS SAME NUMBER OF RESIDUES AS THE PARSED-TOKENISED SEQUENCE FROM MMCIF:
        assert pdb_embed.size(1) == len(aacodes)

        # ONE BACKBONE ATOM (ALPHA-CARBON) PER RESIDUE. SO `len(bbindices)` SHOULD EQUAL NUMBER OF RESIDUES:
        assert len(aacodes) == len(bbindices)

        # MAKE SURE YOU HAVE AT LEAST THE MINIMUM NUMBER OF EXPECTED ATOMS IN MMCIF DATA:
        min_num_atoms_expected_per_residue = 4  # GLYCINE HAS 4 NON-H ATOMS: 1xO, 2xC, 1xN, 5xH.
        min_num_expected_atoms = len(bbindices) * min_num_atoms_expected_per_residue
        # THIS IS THE NUMBER OF ATOMS (AS ONE ROW PER ATOM DUE TO OUTER-JOIN):
        num_of_atoms_in_cif = len(aaindices)

        # ASSUME PROTEIN WILL NEVER BE 100 % GLYCINES (OTHERWISE I'D USE `<=` INSTEAD OF `<`):
        if num_of_atoms_in_cif < min_num_expected_atoms:
            print("WARNING: Too many missing atoms in ", target_pdbid, len(aacodes), len(aaindices))
            continue

        aacodes = np.asarray(aacodes, dtype=np.uint8)
        atomcodes = np.asarray(atomcodes, dtype=np.uint8)
        bbindices = np.asarray(bbindices, dtype=np.int16)
        aaindices = np.asarray(aaindices, dtype=np.int16)
        target_coords = np.asarray(coords, dtype=np.float32)
        # target_coords -= target_coords.mean(0)  # ALREADY DONE IN `parse_tokenise_write_cifs_to_flatfile()`

        assert len(aacodes) == target_coords[bbindices].shape[0]

        # SANITY-CHECKING INTER-ATOMIC DISTANCEs FOR CONSECUTIVE RESIDUES:
        diff = target_coords[1:] - target_coords[:-1]
        distances = np.linalg.norm(diff, axis=1)
        # assert distances.min() >= 0.5, f'Error: Overlapping atoms detected in {target_pdbid}'
        # assert distances.max() <= 3.5, f'Error: Abnormally large gaps in {target_pdbid}'

        print(target_coords.shape, target_pdbid, len(aacodes), distances.min(), distances.max())

        if distances.min() < 1.0 or distances.max() > 2.0:
            abnormal_structures.append((target_pdbid, distances.min(), distances.max()))

        sp.append((aacodes, atomcodes, aaindices, bbindices, target_pdbid, target_coords))

        # Choose every 10th sample for validation
        if tnum % 10 == 0:
            validation_list.append(sp)
        else:
            train_list.append(sp)
        tnum += 1

        # CALCULATE THE STANDARD DEVIATION:
        sum_d2 += (target_coords ** 2).sum()
        sum_d += np.sqrt((target_coords ** 2).sum(axis=-1)).sum()
        nn += target_coords.shape[0]
        sigma_data = sqrt((sum_d2 / nn) - (sum_d / nn) ** 2)
        print(f'Data s.d. = {sigma_data:.2f}')
        print(f'Data unit var scaling = {(1 / sigma_data):.2f}')

    if abnormal_structures:
        print('Abnormal structures detected:')
        for pdbid, min_dist, max_dist in abnormal_structures:
            print(f'{pdbid}: Min {min_dist:.2f}, Max {max_dist:.2f}')

    os.chdir(cwd)  # restore original working directory
    print('Finished `load_dataset()`...')
    return train_list, validation_list


if __name__ == '__main__':

    # # 1. COPY CIFS OVER FROM BIG DATA FOLDER (IF NOT ALREADY DONE FROM DATA_HANDLER.PY):
    # dh.clear_diffdata_mmcif_dir()
    # dh.copy_cifs_from_bigfilefolder_to_diff_data()

    # # 2. CLEAR TOKENISED DIR:
    # dh.clear_diffdata_tokenised_dir()

    from time import time
    start_time = time()

    # 3. PARSE AND TOKENISED CIFS AND WRITE SSV TO TOKENISED DIR:
    _, _lst_file = parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir='../diffusion/diff_data/mmCIF',
                                                         relpath_toknsd_ssv_dir='../diffusion/diff_data/tokenised',
                                                         relpath_pdblst=None,
                                                         flatfile_format_to_write='ssv',
                                                         # pdb_ids=['3C9P'],
                                                         write_lst_file=True)

    # # 4. NOW THAT YOU HAVE A TOKENISED DATASET, YOU CAN PROCEED TO LOAD DATASET AND TRAIN THE MODEL IN TRAINING
    # # SCRIPT `pytorch_protfold_allatomclustrain_singlegpu.py`.
    # # BUT YOU CAN TEST `load_dataset()` FROM HERE IF YOU WANT.
    # # USE A VALID `.lst` FILE THAT ALREADY EXISTS.
    # # EXAMPLE OF SOME TEST FILES:
    # _lst_file = 'pdbchains_9.lst'
    # _lst_file = '3C9P.lst'  # 3C9P has short HETATM stretch near N-term, hence useful for checking `aaindices`.
    # _lst_file = 'globin_1.lst'
    # _lst_file = 'pdbchains_565.lst'
    # # NOTE: AFTER TOKENISING THE 573 SINGLE DOMAIN PROTEINS, `parse_tokenise_write_cifs_to_flatfile()` WRITES A
    # # `.lst` FILE TO `src/diffusion/diff_data/PDBid_list` WITH NAME `pdbchains_<number of tokenised ssv files>.lst`,
    # # HENCE --> 'pdbchains_565.lst').

    # _train_list, _validation_list = load_dataset(f'../diffusion/diff_data/PDBid_list/{_lst_file}')
    time_taken = time() - start_time

    from pathlib import Path
    path = Path('../diffusion/diff_data/mmCIF')
    cif_count = sum(1 for file in path.rglob("*.cif"))
    path = Path('../diffusion/diff_data/tokenised')
    ssv_count = sum(1 for file in path.rglob("*.ssv"))

    print(f'Parsed and tokenised {cif_count} CIFs to SSVs. You have {ssv_count} SSVs. '
          f'This took {time_taken:.2f} seconds in total.')

    # Parsed and tokenised 573 cif files to ssv files. Tokenisation logic, removed 8 of these for not being suitable
    # for potentially different reasons, so there are 565 ssv files saved to `src/diffusion/diff_data/tokenised`.
    # (It took 502.31 seconds in total on my MacBook. It took 283 secs on joe-desktop).
