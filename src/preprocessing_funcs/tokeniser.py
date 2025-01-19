#!~/miniconda3/bin/python
"""
TOKENISER.PY
    - READ IN MMCIF FILE(S).
    - CALL `cif_parser.py` TO  PARSE MMCIF FILE(S)
    - TOKENISE MMCIF
    - WRITE TO .SSV FLATFILE.
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
    Cartn_x             # COORDINATES           - ATOM X-COORDINATES
    Cartn_y             # COORDINATES           - ATOM Y-COORDINATES
    Cartn_z             # COORDINATES           - ATOM Z-COORDINATES
    occupancy           # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.

_pdbx_poly_seq_scheme:
    seq_id              # RESIDUE POSITION      - JOIN TO A_label_seq_id. SORT ON THIS, KEEP IN DATAFRAME.
    mon_id              # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DATAFRAME.
    pdb_seq_num         # RESIDUE POSITION      - KEEP FOR NOW, AS MAY RELATE TO INPUT TO MAKE EMBEDDINGS.
    asym_id             # CHAIN                 - JOIN TO A_label_asym_id, SORT ON THIS, THEN REMOVE IT.

----------------------------------------------------------------------------------------------------------------------
The output of the current `parse_tokenise_cif_write_flatfile()` function is a 12-column dataframe.
'_atom_site' is abbreviated to 'A_' prefix.
'_pdbx_poly_seq_scheme' is abbreviated to 'S_' prefix.
These 13 columns are:

A_label_asym_id       # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
S_seq_id              # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
A_id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
A_label_atom_id       # ATOM                  - KEEP IN DF.
A_Cartn_x             # COORDINATES           - ATOM X-COORDINATES.
A_Cartn_y             # COORDINATES           - ATOM Y-COORDINATES.
A_Cartn_z             # COORDINATES           - ATOM Z-COORDINATES.
aa_label_num          # ENUMERATED RESIDUES   - EQUIVALENT TO `ntcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
bb_or_sc              # BACKBONE OR SIDE-CHAIN ATOM ('bb' or 'sc'), KEEP FOR POSSIBLE SUBSEQUENT OPERATIONS.
bb_atom_pos           # ATOM POSITION CA OR MOST C-TERM OTHER BB ATOM, PER RESIDUE. KEEP IN DF.
bbindices             # INDEX POSITION OF THE ATOM POSITION (`A_id`) OF ALLOCATED BACKBONE ATOM.
atom_label_num        # ENUMERATED ATOMS      - EQUIVALENT TO `atomcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
"""


import os
import re
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api


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
    # THIS IS AN EXTRA VALIDATION STEP THAT IS NOT NEEDED FOR THE 565 PROTEINS, BUT WOULD BE USEFUL FOR FUTURE
    # WILL CHECK THAT EACH NUMERIC COLUMN HAS VALUES IN THE EXPECTED RANGE.
    pass


def _enumerate_residues_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: THIS IS CURRENTLY NOT BEING USED.
    Enumerate residue-atom pairs of mmCIF for one protein, by mapping via pre-written json at `data/enumeration`. Add
    this enumeration to a new column `aa_atom_label_num`. It serves as the tokenised form of polypeptide residue-atom
    pairs for this protein, to be read later to `aaatomcodes` array.
    :param pdf: Dataframe of one protein mmCIF, containing atoms to enumerate to new column.
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
    :param pdf: Dataframe of one protein mmCIF, containing atoms to enumerate to new column.
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
    Enumerate residues, atoms, and residue-atoms pairs of given protein mmCIF data, and store to new columns in given
    dataframe. Currently hard-coded to only use atom data that lacks all hydrogen atoms.
    :param pdfs: List of dataframes of one protein mmCIF, per chain, containing atoms to enumerate to new columns.
    :return: Dataframe with new columns of enumerated data for residues, atoms, and residue-atoms pairs.
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: enumerate atoms and residues')
    result_pdfs = list()
    for pdf in pdfs:
        pdf = _enumerate_residues(pdf)
        pdf = _enumerate_atoms(pdf)
        # pdf = _enumerate_residues_atoms(pdf)  # CURRENTLY NOT BEING USED.
        result_pdfs.append(pdf)
    return result_pdfs


def _assign_backbone_position_to_all_residue_rows(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
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

                # IF NO BACKBONE ATOMS FOR THIS RESIDUE AT ALL, REMOVE THIS RESIDUE FROM THIS MMCIF:
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
    NOTE: CURRENTLY, THIS FUNCTION IS ONLY PERFORMING A SIMPLE CHAIN SELECTION LOGIC, NAMELY TO KEEP THE FIRST IN THE
    GIVEN LIST. FUTURE IMPLEMENTATION MAY LOOK AT KEEPING CHAINS THAT HAVE LESS THAN SOME GIVEN (E.G. 30 %) IDENTITY.
    :param pdfs: Dataframes, one per chain, for given protein mmCIF.
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
    Remove chain(s) from list of chains for this mmCIF if does not have more than a specified number of backbone
    polypeptides.
    :param pdfs: List of dataframes, one per chain, for one mmCIF.
    :param pdb_id: The PDBid of the corresponding mmCIF data.
    :return: List of dataframes, one per chain of protein, with any chains removed if they have less than the minimum
    permitted ratio of missing atoms (arbitrarily chosen for now).
    """
    # If more than this proportion of residues have no backbone atoms, remove the chain.
    MIN_RATIO_MISSING_BACKBONE_ATOMS = 0.0

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
    Remove chain(s) from list of chains for this mmCIF if not polypeptide. Takes advantage of output of previous function
    which assigns to a new column called `bb_or_sc` 'bb' for polypeptide backbone atoms, 'sc' for polypeptide
    side-chain atoms, otherwise `pd.NaT` which therefore indicates the chain is not polypeptide.
    (RCSB mmCIF assigns a different chain to different molecule in a complex, e.g. RNA-protein complex, so if one row in
    the aforementioned column is `pd.NaT`, you should find that all are `pd.NaT`).
    :param pdfs: List of dataframes, one per chain, for one mmCIF.
    :param pdb_id: The PDBid of the corresponding mmCIF data.
    :return: List of dataframes, one per chain of protein, with any non-polypeptide chains removed.
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: remove non-protein chains')
    result_pdfs = []
    for pdf in pdfs:
        try:
            chain = pdf['S_asym_id'].iloc[0]
            atleast_one_row_isna = pdf['bb_or_sc'].isna().any()
            all_rows_isna = pdf['bb_or_sc'].isna().all()
            if atleast_one_row_isna:
                nat_indices = pdf[pd.isna(pdf['bb_or_sc'])].index
                print(f'nat_indices={nat_indices}')
                print(f'There are atoms in chain={chain} of PDBid={pdb_id} which are not polypeptide atoms, so '
                      f'this chain will be excluded.')
                if not all_rows_isna:
                    print(f"It seems that while at least one row in column 'bb_or_sc' has null, "
                          f'not all rows are null. This is unexpected and should be investigated further. '
                          f'(Chain {chain} of PDBid {pdb_id}).')
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
    Fetch raw mmCIF data from API (expected hosted at 'https://files.rcsb.org/download/') using given PDBid,
    and write out to flat file.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :param relpath_dst_cif: Relative path, filename and `.cif` extension of mmCIF to write.
    """
    response = api.call_for_cif_with_pdb_id(pdb_id)
    with open(relpath_dst_cif, 'w') as file:
        file.write(response.text)


def _get_mmcif_data(pdb_id: str, relpath_cif_dir: str) -> dict:
    """
    Read in mmCIF file of specified PDBid, to Python dict. mmCIF file of specified PDBid will be read
    from the specified relative path, if it has previously been downloaded to there, otherwise it will be downloaded
    directly from the rcsb server.
    :param pdb_id: PDBid of mmCIF file to read in.
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
        print(f'Did not find this mmCIF locally ({relpath_cif}). '
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
      - Relative path of directory containing pre-downloaded mmCIF files, e.g. 'diff_data/mmCIF', reading all cif files.
      - Relative path of a list file of PDBids e.g. 'diff_data/SD_573.lst'.
      - One PDBid or a Python list of PDBids.
    "Parsing" involves extracting and casting types of required fields from mmCIF files to dataframes.
    "Tokenising" involves Removing hydrogens, assigning backbone atoms, selecting a suitable chain and enumerating
    atoms and residues.
    Where an mmCIF has > 1 chain, parse & tokenise all polypeptide-only chains & only those with sufficient backbone
    coordinates, according to some pre-decided threshold constant (i.e. `MIN_RATIO_MISSING_BACKBONE_ATOMS`).
    :param relpath_cif_dir: Relative path of directory containing pre-downloaded raw mmCIF files. Has default.
    :param relpath_toknsd_ssv_dir: Relative path of directory for tokenised mmCIFs as flat files. Has default.
    :param relpath_pdblst: <OPTIONAL> Relative path of list file, that lists one or more PDBids or PDBid_chains names.
    e.g. '../diffusion/diff_data/PDBid_list/pdbchains_9.lst'.
    :param flatfile_format_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param pdb_ids: <OPTIONAL> PDBid(s) or PDBid_chain to parse & tokenised. Expect either string of PDBid or list
    of PDBids strings.
    Use 'src/diffusion/diff_data/tokenised' by default, (hence expecting cwd to be 'src/diffusion').
    :param write_lst_file: True to write out a new PDBids_chain `.lst` file of all those that have been parsed and
    tokenised. It is subsequently passed (via its path & file name) to `main()` function in
    `pytorch_protfold_allatomclustrain_singlegpu.py` script.
    :return: List of dataframes, one per chain. Each dataframe has the parsed and tokenised data of one cif file
     and each is also written to a flatfile (ssv by default) at 'src/diffusion/diff_data/tokenised'.
    Dataframe currently has these 13 Columns: ['A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x',
    'A_Cartn_y', 'A_Cartn_z', 'aa_label_num', 'bb_or_sc', 'bb_atom_pos', 'atom_label_num', 'aa_atom_tuple',
    'aa_atom_label_num'].
    """
    # INIT LIST OF PARSED PDB DATA IN DATAFRAMES (ONE DATAFRAME PER PDB AND PER CHAIN):
    cif_pdfs_per_chain = []  # THIS IS RETURNED.

    # THIS IS JUST A LIST OF PDBID_CHAIN (MADE UP OF COMBO OF ANY IN TOKENISED AND NEWLY PARSED):
    for_pdbchain_lst = []

    # BUILD LIST OF PDBIDS TO PROCESS, FROM EITHER GIVEN STRING, LIST, .LST FILE OR PATH TO MMCIFS:
    if pdb_ids:
        if isinstance(pdb_ids, str):  # If string, it is assumed that this is one PDBid.
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
    assert pdb_ids is not None, 'This is a bug! `pdb_ids` is None, but it should be a list of at least one PDBid.'
    assert len(pdb_ids) > 0, 'This is a bug! `pdb_ids` is an empty list, but it should be a list of at least one PDBid.'
    assert len(pdb_ids) > 0, 'This is a bug! `pdb_ids` is an empty list, but it should be a list of at least one PDBid.'

    # MAKE A LIST OF PDBIDS THAT ARE SSVS IN TOKENISED DIR (I.E. HAVE ALREADY BEEN TOKENISED):
    cif_tokenised_ssv_dir = '../diffusion/diff_data/tokenised'
    pdbid_chains_of_pretokenised = []
    if os.path.exists(cif_tokenised_ssv_dir):
        pdbid_chains_of_pretokenised = [item.removesuffix('.ssv')
                                        for item in os.listdir(cif_tokenised_ssv_dir)
                                        if item.endswith('.ssv')]

    for pdb_id in pdb_ids:  # Expecting only one PDBid per line. (Can be PDBid_chain or just PDBid.)
        pdb_id = pdb_id.rstrip().split()[0]
        flatfile_format_to_write = flatfile_format_to_write.removeprefix('.').lower()
        pdb_id = pdb_id.removesuffix('.cif')
        print(f'Starting PDBid={pdb_id} ---------------------------------------------------------')
        if pdb_id == '5TJ5':  # '5TJ5' is one of the 573 single-domain proteins dataset. Its data found to be unusable.
            print(f'{pdb_id} is known to have 2500 missing entries in the aa sequence field. So {pdb_id} will be '
                  f'excluded.')
            continue

        # THE LOGIC EXPECTS THAT IF IT HAS BEEN TOKENISED AND SAVED AS ssv, THEN IT WILL BE BY THE CHAIN AND SO
        # MUST HAVE THE SUFFIX `<pdbid>_A.ssv`, `<pdbid>_B.ssv`, ETC, RATHER THAN JUST `<pdbid>.ssv`:
        if pdb_id in pdbid_chains_of_pretokenised:
            pdbid_chain = pdb_id  # to make it clear that this is expected to include a chain suffix.
            print(f'PDBid={pdbid_chain} already tokenised. Reading ssv in to pdf. Add to list.')
            _cif_pdfs_per_chain = dh.read_tokenised_cif_ssv_to_pdf(relpath_tokensd_dir=relpath_toknsd_ssv_dir,
                                                                   pdb_id=pdbid_chain)
            # FURTHERMORE IT IS EXPECTED THERE IS *ONLY ONE CHAIN* PER PDBID, SO IT IS A PYTHON LIST OF ONE PDB_ID.
            assert _cif_pdfs_per_chain, 'no cif_pdfs_per_chain returned. Expected list of one pdf for one chain.'
            for_pdbchain_lst.append(pdbid_chain)
            continue

        # OTHERWISE GET THE MMCIF (LOCALLY OR VIA API), AND EXPECT PDBid TO BE WITHOUT THE CHAIN:
        pdbid, chain = __split_pdbid_chain(pdb_id)  # Note: chain should be empty string
        mmcif_dict = _get_mmcif_data(relpath_cif_dir=relpath_cif_dir, pdb_id=pdbid)

        # ** AND FINALLY, YOU CAN START PARSING THE MMCIFS (THAT HAVE NOT ALREADY BEEN PARSED) **
        # PARSE MMCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT & JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        # (THIS RETURNS A LIST OF DATAFRAMES, ONE PER POLYPEPTIDE CHAIN).
        cif_pdfs_per_chain = parser.parse_cif(mmcif_dict=mmcif_dict, pdb_id=pdbid)
        if chain:
            # IF A SPECIFIC CHAIN HAS BEEN SPECIFIED, THEN USE ONLY THAT ONE:
            cif_pdfs_per_chain = _keep_only_the_given_chain(pdfs=cif_pdfs_per_chain, chain=chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _remove_all_hydrogen_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _make_new_bckbone_or_sdchain_col(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_of_polypeptide(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_with_enough_bb_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _select_chains_to_use(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)

        if len(cif_pdfs_per_chain) == 0:
            print(f'No chains left for this mmCIF: PDBid={pdbid}, so it cannot be used.')
            continue

        cif_pdfs_per_chain = _assign_backbone_position_to_all_residue_rows(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _enumerate_atoms_and_residues(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        dh.write_tokenised_cifs_to_flatfiles(pdfs=cif_pdfs_per_chain,
                                             dst_data_dir=relpath_toknsd_ssv_dir,
                                             flatfiles=flatfile_format_to_write, pdb_id=pdbid)
        # ADD CHAIN TO NAME OF PDBID:
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


if __name__ == '__main__':

    # # 1. COPY MMCIFS OVER FROM BIG DATA FOLDER (IF NOT ALREADY DONE FROM DATA_HANDLER.PY):
    # dh.clear_diffdata_mmcif_dir()
    # dh.copy_cifs_from_bigfilefolder_to_diff_data()

    # # 2. CLEAR TOKENISED DIR:
    dh.clear_diffdata_tokenised_dir()
    from time import time
    start_time = time()

    # 3. PARSE AND TOKENISED MMCIFS AND WRITE SSV TO TOKENISED DIR:
    _, _lst_file = parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir='../diffusion/diff_data/mmCIF',
                                                         relpath_toknsd_ssv_dir='../diffusion/diff_data/tokenised',
                                                         relpath_pdblst=None,
                                                         flatfile_format_to_write='ssv',
                                                         # pdb_ids=['3C9P'],
                                                         write_lst_file=True)

    time_taken = time() - start_time

    from pathlib import Path
    path = Path('../diffusion/diff_data/mmCIF')
    cif_count = sum(1 for file in path.rglob("*.cif"))
    path = Path('../diffusion/diff_data/tokenised')
    ssv_count = sum(1 for file in path.rglob("*.ssv"))

    print(f'Parsed and tokenised {cif_count} mmCIFs to ssv files. You have {ssv_count} ssv files. '
          f'This took {time_taken:.2f} seconds in total.')

    # Parsed and tokenised 573 cif files to ssv files. Tokenisation logic, removed 8 of these for not being suitable
    # for potentially different reasons, so there are 565 ssv files saved to `src/diffusion/diff_data/tokenised`.
    # (It took 502.31 seconds in total on my MacBook. It took 283 secs on joe-desktop).
