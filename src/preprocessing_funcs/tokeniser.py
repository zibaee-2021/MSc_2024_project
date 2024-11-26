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
bb_index              # POSITION OF THE ALPHA-CARBON FOR EACH RESIDUE IN THE POLYPEPTIDE (MAIN-CHAIN). KEEP IN DF.
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
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import api_caller as api
from src.enums import ColNames, CIF, PolypeptideAtoms
# If more than this proportion of residues have no backbone atoms, remove the chain.
MIN_RATIO_MISSING_BACKBONE_ATOMS = 0.0


# `rp_` stands for relative path:
class Path(Enum):
    rp_diffdata_tokenised_dir = '../diffusion/diff_data/tokenised'
    rp_diffdata_cif_dir = '../diffusion/diff_data/mmCIF'
    rp_diffdata_sd573_lst = '../diffusion/diff_data/SD_573.lst'
    rp_diffdata_globins10_lst = '../diffusion/diff_data/globins_10.lst'
    rp_diffdata_globin1_lst = '../diffusion/diff_data/globin_1.lst'
    rp_diffdata_pdbid_dir = '../diffusion/diff_data/PDBid_list'


class Filename(Enum):
    aa_atoms_no_h = 'residues_atoms_no_hydrogens'
    atoms_no_h = 'unique_atoms_only_no_hydrogens'
    aa = 'residues'


class FileExt(Enum):
    dot_CIF = '.cif'
    ssv = 'ssv'
    dot_ssv = '.ssv'
    dot_lst = '.lst'


class ColValue(Enum):
    bb = 'bb'  # backbone
    sc = 'sc'  # side-chain


def _find_and_parse_missing_data(pdf):

    counts = {
        'NaN': (pdf.map(lambda x: isinstance(x, float) and pd.isna(x))).sum().sum(),
        'pd.NA': (pdf.map(lambda x: x is pd.NA)).sum().sum(),
        'pd.NaT': (pdf.map(lambda x: x is pd.NaT)).sum().sum()
    }
    print(counts)
    has_missing_data = any(value > 0 for value in counts.values())

    if has_missing_data:
        raise ValueError('There are missing values.. needs to be addressed.')

    # missing_strings = ['NaN', 'None', 'N/A', 'missing', 'NULL', '']
    # pdf = pdf.replace(missing_strings, np.nan)
    return pdf


def _assign_mean_corrected_coordinates(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: calc mean-corrected coords')
    result_pdfs = list()
    for pdf in pdfs:
        # SUBTRACT EACH COORDINATE BY THE MEAN OF ALL 3 PER ATOM:
        pdf.loc[:, ColNames.MEAN_COORDS.value] = pdf[[CIF.A_Cartn_x.value,
                                                      CIF.A_Cartn_y.value,
                                                      CIF.A_Cartn_z.value]].mean(axis=1)
        pdf.loc[:, ColNames.MEAN_CORR_X.value] = pdf[CIF.A_Cartn_x.value] - pdf[ColNames.MEAN_COORDS.value]
        pdf.loc[:, ColNames.MEAN_CORR_Y.value] = pdf[CIF.A_Cartn_y.value] - pdf[ColNames.MEAN_COORDS.value]
        pdf.loc[:, ColNames.MEAN_CORR_Z.value] = pdf[CIF.A_Cartn_z.value] - pdf[ColNames.MEAN_COORDS.value]
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
    residues_atoms_enumerated = dh.read_enumerations_json(fname=Filename.aa_atoms_no_h.value)
    # CAST THE STRING REPRESENTATION OF A TUPLE TO AN ACTUAL TUPLE FOR KEY TO WORK IN MAPPING:
    residues_atoms_enumerated = {eval(k): v for k, v in residues_atoms_enumerated.items()}
    # FIRST MAKE NEW COLUMN OF RESIDUE-ATOM PAIRS. E.G. CONTAINS ('ASP':'C'), ('ASP':'CA'), ETC:
    pdf[ColNames.AA_ATOM_PAIR.value] = list(zip(pdf[CIF.S_mon_id.value],
                                                pdf[CIF.A_label_atom_id.value]))

    # MAKE NEW COLUMN FOR ENUMERATED RESIDUE-ATOM PAIRS, VIA RESIDUE-ATOM PAIRS:
    pdf[ColNames.AA_ATOM_LABEL_NUM.value] = (pdf[ColNames.AA_ATOM_PAIR.value]
                                             .map(residues_atoms_enumerated)
                                             .astype('Int64'))
    expected_num_of_cols = 14
    assert len(pdf.columns) == expected_num_of_cols, \
        f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
    return pdf


def _enumerate_atoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate atoms of CIF for one protein, by mapping via pre-written json at `data/enumeration`. Add
    this enumeration to a new column `atom_label_num`. It serves as the tokenised form of polypeptide atoms for this
    protein, to be read later to `atomcodes` array.
    :param pdf: Dataframe of one protein CIF, containing atoms to enumerate to new column.
    :return: Given dataframe with one new column holding the enumerated atoms data. Expected to have 12 columns.
    """
    # MAKE NEW COLUMN FOR ENUMERATED ATOMS ('C', 'CA', ETC), USING JSON->DICT, CAST TO INT:
    atoms_enumerated = dh.read_enumerations_json(fname=Filename.atoms_no_h.value)
    pdf[ColNames.ATOM_LABEL_NUM.value] = (pdf[CIF.A_label_atom_id.value]
                                          .map(atoms_enumerated)
                                          .astype('Int64'))
    expected_num_of_cols = 12
    assert len(pdf.columns) == expected_num_of_cols, \
        f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
    return pdf


def _enumerate_residues(pdf: pd.DataFrame) -> pd.DataFrame:
    # MAKE NEW COLUMN FOR ENUMERATED RESIDUES, USING JSON->DICT, CAST TO INT.
    # `residues_enumerated` DICT KEY AND `S_mon_id` COLUMN VALUES MAP VIA 3-LETTER RESIDUE NAMES:
    residues_enumerated = dh.read_enumerations_json(fname=Filename.aa.value)
    pdf.loc[:, ColNames.AA_LABEL_NUM.value] = (pdf[CIF.S_mon_id.value]
                                                   .map(residues_enumerated)
                                                   .astype('Int64'))
    # pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(residues_enumerated).astype('Int64')
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
        chain = pdf[CIF.S_asym_id.value].unique()[0]

        # Even though `a_id` is always an int (when no NaNs), and I cast to Int64 below. The column remains float64.
        # Apparently I have to cast this column beforehand ? Very odd behaviour.
        pdf[ColNames.BB_INDEX.value] = pd.Series(dtype="Int64")

        for S_seq_id, aa_group in pdf.groupby(CIF.S_seq_id.value):  # GROUP BY RESIDUE POSITION VALUE
            # GET ATOM INDEX ('A_id') WHERE ATOM ('A_label_atom_id') IS 'CA' IN THIS RESIDUE GROUP.
            a_id_of_CA = aa_group.loc[aa_group[CIF.A_label_atom_id.value] == CIF.ALPHA_CARBON.value, CIF.A_id.value]

            # IF NO 'CA' FOR THIS RESIDUE, USE MOST C-TERMINAL NON-CA BACKBONE ATOM POSITION INSTEAD:
            if a_id_of_CA.empty:
                print(f"No 'CA' for {aa_group[CIF.S_mon_id.value].iloc[0]} at {S_seq_id} "
                      f"(PDBid={pdb_id}, chain={chain})")
                positions_of_all_bb_atoms = aa_group.loc[aa_group[ColNames.BB_OR_SC.value]
                                                         == ColValue.bb.value, CIF.A_id.value].to_numpy()

                # IF NO BACKBONE ATOMS FOR THIS RESIDUE AT ALL, REMOVE THIS RESIDUE FROM THIS CIF:
                if positions_of_all_bb_atoms.size == 0:
                    aa = {aa_group[CIF.S_mon_id.value].iloc[0]}
                    print(f'{aa} at {S_seq_id} only has atoms {str(list(aa_group[CIF.A_label_atom_id.value]))}. '
                          f'Hence, no backbone atoms at all, so {aa} at {S_seq_id} will be completely removed from '
                          f'this dataframe.')
                    pdf = pdf[pdf[CIF.S_seq_id.value] != S_seq_id]
                    continue  # continue to next residue
                else:
                    a_id = max(positions_of_all_bb_atoms)
                    print(f'Instead, assigning position of most C-terminal non-CA backbone atom={a_id}.')
                    most_cterm_bb_atom = aa_group.loc[aa_group[CIF.A_id.value]
                                                      == a_id, CIF.A_label_atom_id.value].values[0]
                    print(f'Non-CA backbone atoms for this residue are at: {str(list(positions_of_all_bb_atoms))}, '
                          f'so {a_id} is selected. (The atom at this position is: {most_cterm_bb_atom}.)')
                    continue
                # raise ValueError(f'No {CIF.ALPHA_CARBON.value} found in {CIF.A_label_atom_id.value} for group '
                #                  f'{group[CIF.S_seq_id.value].iloc[0]}')
            else:
                a_id = a_id_of_CA.iloc[0]

                # ASSIGN THIS ATOM INDEX TO BB_INDEX ('bb_index') FOR ALL ROWS IN THIS GROUP:
            pdf.loc[aa_group.index, ColNames.BB_INDEX.value] = a_id

        # CAST NEW COLUMN TO INT64 (FOR CONSISTENCY):
        print(f'What type is pdf bb_index column before any casting operations ..{pdf[ColNames.BB_INDEX.value].dtype}')

        pdf.loc[:, ColNames.BB_INDEX.value] = pd.to_numeric(pdf[ColNames.BB_INDEX.value], errors='coerce')
        print(f'pdf bb_index column should be numeric type ..{pdf[ColNames.BB_INDEX.value].dtype}')

        pdf.loc[:, ColNames.BB_INDEX.value] = pdf[ColNames.BB_INDEX.value].astype('Int64')
        print(f'pdf bb_index column should be integer type ..{pdf[ColNames.BB_INDEX.value].dtype}')
        result_pdfs.append(pdf)

    return result_pdfs


def _select_chains_to_use(pdfs: List[pd.DataFrame], chains: list=None, pdb_id: str=None) -> List[pd.DataFrame]:
    """
    Select which chains to keep for further parsing and tokenisation and to be written to flatfile. If no chains
    specified, all protein chains will be kept. If no PDBid is given, don't print anything out.
    :param pdfs: Dataframes, one per chain, for given protein CIF.
    :param chains: One of more chain(s) to keep. e.g. [A, C].
    :param pdb_id: Just for printing out as part of tracking the function calls.
    :return: A list of one or more dataframes, according to which chains to keep (This is also temporary to avoid
    breaking subsequent
    operations).
    """
    if pdb_id:
        print(f'PDBid={pdb_id}: select which chains to keep.')
    if len(pdfs) > 0:
        pdfs = [pdfs[0]]
    return pdfs


def _only_keep_chains_with_enuf_bckbone_atoms(pdfs: List[pd.DataFrame], pdb_id: str=None) -> List[pd.DataFrame]:
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
                       .groupby(CIF.S_seq_id.value)[ColNames.BB_OR_SC.value]
                       .apply(lambda x: ColValue.bb.value not in x.values)
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
            chain = pdf[CIF.S_asym_id.value].iloc[0]
            atleast_one_row_isna = pdf[ColNames.BB_OR_SC.value].isna().any()
            all_rows_isna = pdf[ColNames.BB_OR_SC.value].isna().all()
            if atleast_one_row_isna:
                nat_indices = pdf[pd.isna(pdf[ColNames.BB_OR_SC.value])].index
                print(f'nat_indices={nat_indices}')
                print(f'There are atoms in chain={chain} of PDB id={pdb_id} which are not polypeptide atoms, so this chain '
                      f'will be excluded.')
                if not all_rows_isna:
                    print(f'It seems that while at least one row in column {ColNames.BB_OR_SC.value} has null, '
                          f'not all rows are null. This is unexpected and should be investigated further. '
                          f'(Chain {chain} of PDB id {pdb_id}).')
            else:
                result_pdfs.append(pdf)
            if len(result_pdfs) == 0:
                print(f'PDBid={pdb_id}: After removing all non-polypeptide chains, there are no chains left. '
                      f'This should not occur, so needs to be investigated further.')
        except IndexError:
            print(f' `chain = pdf[CIF.S_asym_id.value].iloc[0]` fails. '
                  f'\nCIF.S_asym_id.value={CIF.S_asym_id.value} '
                  f'\npdf[CIF.S_asym_id.value]={pdf[CIF.S_asym_id.value]}')
    return result_pdfs


def _make_new_bckbone_or_sdchain_col(pdfs: List[pd.DataFrame], pdb_id: str=None) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: make new column `bb_or_sc` - indicates whether atom is backbone or side-chain.')

    result_pdfs = list()

    for pdf in pdfs:
        is_backbone_atom = pdf[CIF.A_label_atom_id.value].isin(PolypeptideAtoms.BACKBONE.value)
        is_sidechain_atom = pdf[CIF.A_label_atom_id.value].isin(PolypeptideAtoms.SIDECHAIN.value)

        # MAKE NEW COLUMN TO INDICATE IF ATOM IS FROM POLYPEPTIDE BACKBONE ('bb) OR SIDE-CHAIN ('sc'):
        pdf.loc[:, ColNames.BB_OR_SC.value] = np.select([is_backbone_atom, is_sidechain_atom],
                                                        [ColValue.bb.value, ColValue.sc.value], default=pd.NaT)
        expected_num_of_cols = 9
        assert len(pdf.columns) == expected_num_of_cols, \
            f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
        result_pdfs.append(pdf)

    return result_pdfs


def _remove_all_hydrogen_atoms(pdfs: List[pd.DataFrame], pdb_id: str=None) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: remove Hydrogens')
    hydrogen_atoms = dh.read_lst_file_from_data_dir(dh.Path.enumeration_h_list.value)
    result_pdfs = []
    for pdf in pdfs:
        _pdf = pdf.loc[~pdf[CIF.A_label_atom_id.value].isin(hydrogen_atoms)]
        result_pdfs.append(_pdf)
    return result_pdfs


def _keep_only_the_given_chain(pdfs: List[pd.DataFrame], chain: str, pdb_id: str=None) -> List[pd.DataFrame]:
    if pdb_id:
        print(f'PDBid={pdb_id}: only keep the specified chain={chain}.')
    result_pdfs = list()
    for pdf in pdfs:
        if pdf[CIF.S_asym_id.value].isin([chain]).any():
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
    # relpath_dst_cif = f'../data/big_data_to_git_ignore/SD_573_CIFs/{pdb_id}.cif'
    with open(relpath_dst_cif, 'w') as file:
        file.write(response.text)


def _get_mmcif_data(pdb_id: str, relpath_cif_dir: str) -> dict:
    """
    :param pdb_id:
    :param relpath_cif_dir:
    :return:
    """
    relpath_cif_dir = relpath_cif_dir.removesuffix(FileExt.dot_CIF.value).removeprefix('/').removesuffix('/')
    pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
    relpath_cif = f'{relpath_cif_dir}/{pdb_id}{FileExt.dot_CIF.value}'
    if os.path.exists(relpath_cif):
        try:
            MMCIF2Dict(relpath_cif)
        except ValueError:
            print(f'{pdb_id}{FileExt.dot_CIF.value} appears to be empty. '
                  f'Attempt to read {pdb_id} directly from {api.Urls.PDB.value}/{pdb_id}')
            __fetch_mmcif_from_pdb_api_and_write(pdb_id=pdb_id, relpath_dst_cif=relpath_cif)
    else:
        print(f'Did not find this CIF locally ({relpath_cif}). '
              f'Attempt to read {pdb_id} directly from {api.Urls.PDB.value}/{pdb_id}')
        __fetch_mmcif_from_pdb_api_and_write(pdb_id=pdb_id, relpath_dst_cif=relpath_cif)
    mmcif = MMCIF2Dict(relpath_cif)
    if not mmcif:
        print(f'{relpath_cif}/{pdb_id}{FileExt.dot_CIF.value} appears to be empty. ')
    return mmcif


def __split_pdbid_chain(pdbid_chain):
    match = re.match(r"^(.*)_([A-Za-z])$", pdbid_chain)
    if match:
        pdbid, chain = match.groups()
        return pdbid, chain
    else:
        return pdbid_chain, None


def parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir=Path.rp_diffdata_cif_dir.value,
                                          relpath_toknsd_ssv_dir=Path.rp_diffdata_tokenised_dir.value,
                                          relpath_pdblst: str = None,
                                          flatfile_format_to_write: str = FileExt.ssv.value,
                                          pdb_ids=None,
                                          write_lst_file=False) -> List[pd.DataFrame]:
    """
    Parse and tokenise sequence and structure fields from mmCIFs of proteins, returning a list of Pandas dataframes,
    with one dataframe per chain. Write each dataframe to one flat file (ssv by default), in `tokenised` directory,
    The mmCIFs to parse, tokenise and write to flat flats is specified by one or more of the following 3:
      - Relative path of directory containing pre-downloaded CIF files, e.g. `diff_data/mmCIF`, reading all CIFs in.
      - Relative path of a list file of PDB ids e.g. `diff_data/SD_573.lst`.
      - A Python list of PDB ids.
    "Parsing" involves extracting required fields from mmCIF files to dataframes.
    "Tokenising" involves enumerating atoms and residues, and calculating mean-adjusted x, y, z coordinates.
    Where a CIF has > 1 chain, parse & tokenise all polypeptide-only chains & only those with sufficient backbone
    coordinates, according to some pre-decided threshold constant (i.e. `MIN_RATIO_MISSING_BACKBONE_ATOMS`).
    :param relpath_cif_dir: Relative path of directory containing pre-downloaded raw mmCIF files.
    :param relpath_toknsd_ssv_dir: Relative path of directory for tokenised CIFs as flat files.
    :param relpath_pdblst: <OPTIONAL> Relative path to list file of one or more PDB ids or PDBid_chains names. e.g.
    `globins_10.lst`.
    :param flatfile_format_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param pdb_ids: <OPTIONAL> PDB id(s) to parse & tokenised. Expected as a string of one PDB id or a list of PDB ids.
    Uses `src/diffusion/diff_data/mmCIF` subdir by default, because expecting call from `src/diffusion`.
    Use `src/diffusion/diff_data/tokenised` by default, because expecting call from `src/diffusion`.
    :return: Parsed and tokenised CIF file as dataframe which is also written to a flatfile (ssv by default)
    at `src/diffusion/diff_data/tokenised`. List of dataframes, one per chain.
    Dataframe currently has these 17 Columns: ['A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x',
    'A_Cartn_y', 'A_Cartn_z', 'aa_label_num', 'bb_or_sc', 'bb_index', 'atom_label_num', 'aa_atom_tuple',
    'aa_atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z'].
    """
    for_pdbchain_lst = []

    if pdb_ids:
        if isinstance(pdb_ids, str):  # If string, it is assumed that this is one PDB id.
            pdb_ids = [pdb_ids]
    if relpath_pdblst:
        pdb_ids.extend(dh.read_pdb_lst_from_src_diff_dir(relpath_pdblst=relpath_pdblst))

    cif_tokenised_ssv_dir = Path.rp_diffdata_tokenised_dir.value
    pdbid_chains_of_pretokenised = []
    if os.path.exists(cif_tokenised_ssv_dir):
        pdbid_chains_of_pretokenised = [item.removesuffix(FileExt.dot_ssv.value)
                                             for item in os.listdir(cif_tokenised_ssv_dir)
                                             if item.endswith(FileExt.dot_ssv.value)]
    cif_pdfs_per_chain = []

    for pdb_id in pdb_ids:  # Expecting only one PDB id per line. Can be PDBid_chain.
        pdb_id = pdb_id.rstrip().split()[0]
        flatfile_format_to_write = flatfile_format_to_write.removeprefix('.').lower()
        pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
        print(f'Starting PDBid={pdb_id} ---------------------------------------------------------')

        # THE LOGIC EXPECTS THAT IF IT HAS BEEN TOKENISED AND SAVED AS ssv, THEN IT WILL BE BY THE CHAIN AND SO
        # MUST HAVE THE SUFFIX `<pdbid>_A.ssv`, `<pdbid>_B.ssv`, ETC, RATHER THAN JUST `<pdbid>.ssv`:
        if pdb_id in pdbid_chains_of_pretokenised:
            pdbid_chain = pdb_id  # to make it clear that this is expected to include a chain suffix.
            print(f'PDBid={pdbid_chain} already tokenised. Reading ssv in to pdf. Add to list.')
            _cif_pdfs_per_chain = dh.read_tokenised_cif_ssv_to_pdf(relpath_tokensd_dir=relpath_toknsd_ssv_dir,
                                                                   pdb_id=pdbid_chain)
            # EXPECTED TO BE FOR ONE CHAIN OF THIS PDBID, SO IT'S A PYTHON LIST OF 1.
            assert _cif_pdfs_per_chain, 'no cif_pdfs_per_chain returned. Expected list of one pdf for one chain.'
            for_pdbchain_lst.append(pdbid_chain)
            continue
        # OTHERWISE GET THE CIF (LOCALLY OR VIA API), AND EXPECT PDBid TO BE WITHOUT THE CHAIN:
        relpath_cif_dir = relpath_cif_dir.removesuffix('/').removeprefix('/')
        pdbid, chain = __split_pdbid_chain(pdb_id)  # Note: chain should be empty string
        mmcif_dict = _get_mmcif_data(relpath_cif_dir=relpath_cif_dir, pdb_id=pdbid)
        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT & JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        # (THIS RETURNS A LIST OF DATAFRAMES, ONE PER POLYPEPTIDE CHAIN).
        cif_pdfs_per_chain = parser.parse_cif(mmcif_dict=mmcif_dict, pdb_id=pdbid)
        if chain:
            cif_pdfs_per_chain = _keep_only_the_given_chain(pdfs=cif_pdfs_per_chain, chain=chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _remove_all_hydrogen_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _make_new_bckbone_or_sdchain_col(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_of_polypeptide(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _only_keep_chains_with_enuf_bckbone_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _select_chains_to_use(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)

        if len(cif_pdfs_per_chain) == 0:
            print(f'No chains left for this CIF: PDBid={pdbid}, so it cannot be used.')
            continue

        cif_pdfs_per_chain = _assign_backbone_index_to_all_residue_rows(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _enumerate_atoms_and_residues(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = _assign_mean_corrected_coordinates(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        dh.write_tokenised_cif_to_flatfile(pdfs=cif_pdfs_per_chain,
                                           dst_data_dir=relpath_toknsd_ssv_dir,
                                           flatfiles=flatfile_format_to_write, pdb_id=pdbid)
        for pdf_chain in cif_pdfs_per_chain:
            _find_and_parse_missing_data(pdf_chain)
            chain = pdf_chain[CIF.S_asym_id.value].iloc[0]
            pdbid_chain = f'{pdbid}_{chain}'
            for_pdbchain_lst.append(pdbid_chain)

    if write_lst_file:
        dh.write_list_to_lst_file(list_to_write=for_pdbchain_lst,
                                  path_fname=f'{Path.rp_diffdata_pdbid_dir.value}/pdbchains_{len(for_pdbchain_lst)}'
                                             f'{FileExt.dot_lst.value}')
    return cif_pdfs_per_chain


if __name__ == '__main__':
    # dh.copy_cifs_from_bigfilefolder_to_diff_data()

    parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir=Path.rp_diffdata_cif_dir.value,
                                          relpath_toknsd_ssv_dir=Path.rp_diffdata_tokenised_dir.value,
                                          relpath_pdblst=None,
                                          flatfile_format_to_write=FileExt.ssv.value,
                                          pdb_ids=['1ECA', '2DN1', '2DN2', '1OJ6', '1V5H',
                                                   '1MBN', '2GDM', '1GDI', '2WY4'],
                                          write_lst_file=True)
    # Note: '4C0N' has 18 missing values
    # dh.clear_diffdatacif_dir()
