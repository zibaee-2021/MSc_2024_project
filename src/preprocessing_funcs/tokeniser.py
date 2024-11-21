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
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from src.enums import ColNames, CIF, PolypeptideAtoms


class Path(Enum):
    relpath_diffdata_tokenised_dir = '../diffusion/diff_data/tokenised'
    relpath_diffdata_sd573_lst = '../diffusion/diff_data/SD_573.lst'
    relpath_diffdata_cif_dir = '../diffusion/diff_data/mmCIF'


class Filename(Enum):
    aa_atoms_no_h = 'residues_atoms_no_hydrogens'
    atoms_no_h = 'unique_atoms_only_no_hydrogens'
    aa = 'residues'


class FileExt(Enum):
    dot_CIF = '.cif'
    ssv = 'ssv'
    dot_ssv = '.ssv'


class ColValue(Enum):
    bb = 'bb'  # backbone
    sc = 'sc'  # sidechain


def _assign_mean_corrected_coordinates(pdfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
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


def _enumerate_atoms_and_residues(pdfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Enumerate residues, atoms, and residue-atoms pairs of given protein CIF data, and store to new columns in given
    dataframe. Currently hard-coded to only use atom data that lacks all hydrogen atoms.
    :param pdfs: List of dataframes of one protein CIF, per chain, containing atoms to enumerate to new columns.
    :return: Dataframe with new columns of enumerated data for residues, atoms, and residue-atoms pairs.
    """
    result_pdfs = list()
    for pdf in pdfs:
        pdf = _enumerate_residues(pdf)
        pdf = _enumerate_atoms(pdf)
        pdf = _enumerate_residues_atoms(pdf)
        result_pdfs.append(pdf)
    return result_pdfs


def _assign_backbone_index_to_all_residue_rows(pdfs: List[pd.DataFrame], pdb_id: str) -> List[pd.DataFrame]:
    result_pdfs = list()
    for pdf in pdfs:
        # ASSIGN INDEX OF CHOSEN BACKBONE ATOM (ALPHA-CARBON) FOR ALL ROWS IN EACH ROW-WISE-RESIDUE SUBSETS:
        for S_seq_id, group in pdf.groupby(CIF.S_seq_id.value):  # GROUP BY RESIDUE POSITION VALUE
            # GET ATOM INDEX ('A_id') WHERE ATOM ('A_label_atom_id') IS 'CA' IN THIS RESIDUE GROUP.
            a_id_of_CA = group.loc[group[CIF.A_label_atom_id.value] == CIF.ALPHA_CARBON.value, CIF.A_id.value]
            chain = pdf[CIF.S_asym_id.value].unique()[0]
            # CHECK THERE'S AT LEAST ONE 'CA' IN THIS GROUP:
            if a_id_of_CA.empty:
                print(f"No 'CA' for residue {S_seq_id} in {pdb_id}, chain {chain}")
                positions_of_all_bb_atoms = group.loc[group[ColNames.BB_OR_SC.value]
                                                      == ColValue.bb.value, CIF.A_id.value].to_numpy()
                if not positions_of_all_bb_atoms:
                    print(f'No backbone atoms at all in chain={chain} of PDBid={pdb_id}.')
                print(f'Positions of all other backbone atoms for this residue = {positions_of_all_bb_atoms}')
                max_bb_position = max(positions_of_all_bb_atoms)
                print(f'Position chosen of backbone atom={max_bb_position}')
                most_Cterminal_bb_atom = group.loc[group[CIF.A_id.value]
                                                   == max_bb_position, CIF.A_label_atom_id.value].values[0]
                print(f'Therefore, selecting most C-terminal position of another backbone atom for this residue. '
                      f'Non-CA backbone atoms are at {str(list(positions_of_all_bb_atoms))}. '
                      f"So {max_bb_position} is selected. This atom is '{most_Cterminal_bb_atom}'.")
                if positions_of_all_bb_atoms.size == 0:
                    print(f'There are no backbone atoms at all! (Residue {S_seq_id}, pdb {pdb_id}, chain {chain}. '
                          f'Ignoring this for now but a decision needs to be made about whether to remove '
                          f'this residue all together or if it is ok to substitute in the position of a side-chain '
                          f'atom.')
                    continue
                # raise ValueError(f'No {CIF.ALPHA_CARBON.value} found in {CIF.A_label_atom_id.value} for group '
                #                  f'{group[CIF.S_seq_id.value].iloc[0]}')
            else:
                a_id = a_id_of_CA.iloc[0]

                # ASSIGN THIS ATOM INDEX TO BB_INDEX ('bb_index') FOR ALL ROWS IN THIS GROUP:
                pdf.loc[group.index, ColNames.BB_INDEX.value] = a_id
        # CAST NEW COLUMN TO INT64 (FOR CONSISTENCY):
        pdf[ColNames.BB_INDEX.value] = pd.to_numeric(pdf[ColNames.BB_INDEX.value], errors='coerce')
        pdf[ColNames.BB_INDEX.value] = pdf[ColNames.BB_INDEX.value].astype('Int64')
        result_pdfs.append(pdf)
    return result_pdfs


def _make_new_column_for_backbone_or_sidechain_label(pdfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    result_pdfs = list()
    for pdf in pdfs:
        # MAKE NEW COLUMN TO INDICATE IF ATOM IS FROM POLYPEPTIDE BACKBONE ('bb) OR SIDE-CHAIN ('sc'):
        pdf.loc[:, ColNames.BB_OR_SC.value] = np.select([is_backbone_atom, is_sidechain_atom],
                                                        [ColValue.bb.value, ColValue.sc.value], default=pd.NaT)
        expected_num_of_cols = 9
        assert len(pdf.columns) == expected_num_of_cols, \
            f'Dataframe should have {expected_num_of_cols} columns. But this has {len(pdf.columns)}'
        result_pdfs.append(pdf)
    return result_pdfs


def parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir: str, relpath_dst_dir: str,
                                          flatfile_format_to_write: str = FileExt.ssv.value,
                                          pdb_id: str = None) -> List[pd.DataFrame]:
    """
    Parse, then tokenise structure-related information in mmCIF files for proteins as specified by their PDB
    entries (`pdb_ids`) - unique Protein Data Bank identifiers.
    Write the tokenised sequence and structure data to flat files (ssv by default) in `tokenised` subdir.
    Parsing involves extracting required fields from mmCIF files to dataframes.
    Tokenising involves enumerating atoms and residues, and mean-adjusting x, y, z coordinates.
    If the PDB has more than one chain, just use the one chain (or the longest chain??? TODO)
    :param relpath_cif_dir: Relative path to source dir of the raw cif files to be parsed and tokenised.
    :param relpath_dst_dir: Relative path to destination dir for the parsed and tokenised cif as a flat file.
    :param flatfile_format_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param pdb_id: <OPTIONAL> PDB id of a given protein to parse and tokenised. (Used for testing).
    Uses `src/diffusion/diff_data/mmCIF` subdir by default, because expecting call from `src/diffusion`.
    :param relpath_dst_dir: Relative path to destination dir for the parsed and tokenised cif as a flat file.
    Use `src/diffusion/diff_data/tokenised` by default, because expecting call from `src/diffusion`.
    :return: Parsed and tokenised cif file as dataframe which is also written to a flatfile (ssv by default)
    at `src/diffusion/diff_data/tokenised`. List of dataframes, one per chain.
    Dataframe currently has these 17 Columns: ['A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x',
    'A_Cartn_y', 'A_Cartn_z', 'aa_label_num', 'bb_or_sc', 'bb_index', 'atom_label_num', 'aa_atom_tuple',
    'aa_atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z'].
    """
    if pdb_id:
        _pdb_list = [pdb_id]
    else:
        _pdb_list = __read_lst_file_from_src_diff_dir(fname=Path.relpath_diffdata_sd573_lst.value)

    list_of_cif_pdfs_per_chain = []

    for pdb_id in _pdb_list:  # expecting only one PDB id per line.
        pdb_id = pdb_id.rstrip().split()[0]
        flatfile_format_to_write = flatfile_format_to_write.removeprefix('.').lower()
        pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
        # IF ALREADY PARSED AND SAVED AS FLATFILE, JUST READ IT IN:
        cif_tokenised_ssv = f'{relpath_dst_dir}/{pdb_id}_A.{flatfile_format_to_write}'
        if os.path.exists(cif_tokenised_ssv):
            list_of_cif_pdfs_per_chain = dh.read_tokenised_cif_ssv_to_pdf(pdb_id=pdb_id,
                                                                          relpath_tokenised_dir=cif_tokenised_ssv_dir)
        else:
            # OTHERWISE GET THE CIF DATA (EITHER LOCALLY OR VIA API)
            relpath_cif_dir = relpath_cif_dir.removesuffix('/').removeprefix('/')
            # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT AND JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
            # (THIS RETURNS A LIST OF DATAFRAMES, ONE PER POLYPEPTIDE CHAIN).
            list_of_cif_pdfs_per_chain = parser.parse_cif(pdb_id=pdb_id, relpath_cifs_dir=relpath_cif_dir)
            list_of_cif_pdfs_per_chain = _make_new_column_for_backbone_or_sidechain_label(list_of_cif_pdfs_per_chain)
            list_of_cif_pdfs_per_chain = _only_keep_chains_of_polypeptide(list_of_cif_pdfs_per_chain, pdb_id)
            list_of_cif_pdfs_per_chain = _only_keep_chains_with_enuf_backbone_atoms(list_of_cif_pdfs_per_chain, pdb_id)

            # Temporary hack to tokenise & write just 1 chain to ssv: TODO decide how to select which chain to use.

            list_of_cif_pdfs_per_chain = [list_of_cif_pdfs_per_chain[0]]
            list_of_cif_pdfs_per_chain = _make_new_column_for_backbone_or_sidechain_label(list_of_cif_pdfs_per_chain)
            list_of_cif_pdfs_per_chain = _assign_backbone_index_to_all_residue_rows(list_of_cif_pdfs_per_chain, pdb_id)
            list_of_cif_pdfs_per_chain = _enumerate_atoms_and_residues(list_of_cif_pdfs_per_chain)
            list_of_cif_pdfs_per_chain = _assign_mean_corrected_coordinates(list_of_cif_pdfs_per_chain)
            dh.write_tokenised_cif_to_flatfile(pdb_id, list_of_cif_pdfs_per_chain,
                                               dst_data_dir=relpath_dst_dir,
                                               flatfiles=flatfile_format_to_write)
    return list_of_cif_pdfs_per_chain


def __read_lst_file_from_src_diff_dir(fname: str) -> list:
    try:
        with open(fname, 'r') as lst_f:
            pdb_list = [line.strip() for line in lst_f]
    except FileNotFoundError:
        print(f'{lst_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return pdb_list


if __name__ == '__main__':
    # dh.copy_cifs_from_bigfilefolder_to_diff_data()
    parse_tokenise_write_cif_to_flatfile(relpath_cif_dir=Path.relpath_diffdata_cif_dir.value,
                                         relpath_dst_dir=Path.relpath_diffdata_tokenised_dir.value)

    # dh.clear_diffdatacif_dir()
