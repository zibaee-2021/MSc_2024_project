"""
TOKENISER.PY
    - CALLS CIF PARSER TO READ IN AND PARSE THE CIF FILE TO EXTRACT THE FOLLOWING 16 FIELDS. WRITE TO ssv FLATFILE.
    - ENUMERATES ATOMS AND AMINO ACID RESIDUES.
    - SUBTRACTS COORDINATES BY THEIR MEAN COORDINATE VALUES PER ATOM.
----------------------------------------------------------------------------------------------------------------------
The following 14 mmCIF fields are extracted from the raw mmCIF files, parsed and tokenised into a dataframe.
These 14 fields are:

atom_site:
    group_PDB           # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
    label_seq_id         # RESIDUE POSITION      - USED TO JOIN WITH S_pdb_seq_num, THEN REMOVE IT.
    label_comp_id       # RESIDUE (3-LETTER)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
    id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
    label_atom_id       # ATOM                  - KEEP IN DF.
    label_asym_id       # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
    Cartn_x             # ATOM COORDS          - X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_y             # ATOM COORDS           - Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_z             # ATOM COORDS           - Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    occupancy           # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.

_pdbx_poly_seq_scheme:
    seq_id              # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
    mon_id              # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
    pdb_seq_num         # RESIDUE POSITION      - JOIN TO A_label_seq_id, THEN REMOVE IT.
    asym_id             # CHAIN                 - JOIN ON THIS, SORT ON THIS, THEN REMOVE IT.

----------------------------------------------------------------------------------------------------------------------
The output of the current `parse_tokenise_cif_write_flatfile()` function is a 17-column dataframe.
These 17 columns are:

A_label_asym_id       # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
S_seq_id              # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
A_id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
A_label_atom_id       # ATOM                  - KEEP IN DF.
A_Cartn_x             # ATOM COORDS           - X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_y             # ATOM COORDS           - Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_z             # ATOM COORDS           - Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
aa_label_num          # ENUMERATED RESIDUES   - EQUIVALENT TO `ntcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
bb_or_sc              # BACKBONE OR SIDE-CHAIN ATOM ('bb' or 'sc'), KEEP FOR POSSIBLE SUBSEQUENT OPERATIONS.
bb_index              # POSITION OF THE ALPHA-CARBON FOR EACH RESIDUE IN THE POLYPEPTIDE (MAIN-CHAIN). KEEP IN DF.
atom_label_num        # ENUMERATED ATOMS      - EQUIVALENT TO `atomcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
aa_atom_tuple         # RESIDUE-ATOM PAIR     - ONE TUPLE PER ROW. KEEP IN DF.
aa_atom_label_num     # ENUMERATED RESIDUE-ATOM PAIRS. (ALTERNATIVE WAY TO GENERATE `atomcodes`).
mean_xyz              # MEAN OF COORDS        - MEAN OF X, Y, Z COORDINATES FOR EACH ATOM. KEEP IN DF TEMPORARILY.
mean_corrected_x      # X COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_y      # Y COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_z      # Z COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.


"""


import os
import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from src.enums import ColNames, CIF, PolypeptideAtoms


# TODO this function needs a unit test. Make a small cif and make sure it does what it should
def parse_tokenise_cif_write_flatfile(pdb_ids=None, flatfile_format_to_write: str = 'ssv',
                                      relpath_to_raw_cifs_dir='diff_data/cif',
                                      relpath_to_dst_dir='diff_data/tokenised') -> pd.DataFrame:
    """
    Parse, then tokenise structure-related information in mmCIF files for proteins as specified by their PDB
    entries (`pdb_ids`) - unique Protein Data Bank identifiers.
    Write the tokenised sequence and structure data to flat files (ssv by default) in `tokenised` subdir.
    Parsing involves extracting required fields from mmCIF files to dataframes.
    Tokenising involves enumerating atoms and residues, and mean-adjusting x, y, z coordinates.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    :param flatfile_format_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param relpath_to_raw_cifs_dir: Relative path to source dir of the raw cif files to be parsed and tokenised.
    Uses `src/diffusion/diff_data/cif` subdir by default, because expecting call from `src/diffusion`.
    :param relpath_to_dst_dir: Relative path to destination dir for the parsed and tokenised cif as a flat file.
    Use `src/diffusion/diff_data/tokenised` by default, because expecting call from `src/diffusion`.
    :return: Parsed and tokenised cif file as dataframe which is also written to a flatfile (ssv by default)
    at `src/diffusion/diff_data/tokenised`.
    Dataframe currently has these 17 Columns: ['A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x',
    'A_Cartn_y', 'A_Cartn_z', 'aa_label_num', 'bb_or_sc', 'bb_index', 'atom_label_num', 'aa_atom_tuple',
    'aa_atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z'].
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        relpath_to_raw_cifs_dir = relpath_to_raw_cifs_dir.removesuffix('/').removeprefix('/')
        pdb_id = pdb_id.removesuffix('.cif')
        cif = f'{relpath_to_raw_cifs_dir}/{pdb_id}.cif'
        assert os.path.exists(cif)

        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT AND JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, path_to_raw_cif=cif)
        expected_num_of_cols = 8
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # READ ENUMERATION MAPPINGS TO DICTS TO USE FOR CONVERTING RESIDUES AND ATOMS TO NUMBERS:
        # NB: THIS FUNCTION CURRENTLY READS ONLY THE MAPPINGS THAT LACK HYDROGENS:
        residues_atoms_enumerated, atoms_enumerated, residues_enumerated = dh.read_enumeration_mappings()

        # TODO: test this filter step to create new column with 'bb' or 'sc' works or not.
        # MAKE NEW COLUMN TO INDICATE IF ATOM IS FROM POLYPEPTIDE BACKBONE ('bb) OR SIDE-CHAIN ('sc'):
        pdf_cif.loc[:, ColNames.BACKBONE_SIDECHAIN.value] \
            = pdf_cif[CIF.A_label_atom_id.value].isin(PolypeptideAtoms.BACKBONE.value).replace({True: 'bb', False: 'sc'})
        expected_num_of_cols = 9
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # ASSIGN INDEX OF CHOSEN BACKBONE ATOM (ALPHA-CARBON) FOR ALL ROWS PER RESIDUE ROW-WISE SUBSETS:
        for _, group in pdf_cif.groupby(CIF.S_seq_id.value):  # GROUP BY RESIDUE POSITION VALUE
            # GET ATOM INDEX ('A_id') WHERE ATOM ('A_label_atom_id') IS 'CA' IN THIS RESIDUE GROUP.
            a_id_of_CA = group.loc[group[CIF.A_label_atom_id.value] == CIF.ALPHA_CARBON.value, CIF.A_id.value]

            # CHECK THERE'S AT LEAST ONE CA IN THIS GROUP:
            if a_id_of_CA.empty:
                raise ValueError(f'No {CIF.ALPHA_CARBON.value} found in {CIF.A_label_atom_id.value} for group '
                                 f'{group[CIF.S_seq_id.value].iloc[0]}')
            else:
                a_id = a_id_of_CA.iloc[0]

                # ASSIGN THIS ATOM INDEX TO BB_INDEX ('bb_index') FOR ALL ROWS IN THIS GROUP:
                pdf_cif.loc[group.index, ColNames.BB_INDEX.value] = a_id

        # MAKE NEW COLUMN FOR ENUMERATED RESIDUES, USING `aa_atoms_enumerated` JSON->DICT AND CAST TO INT.
        # NB: `residues_enumerated` AND `S_mon_id` ARE BOTH USING 3-LETTER RESIDUE NAMES SO MAPPING IS FINE:
        pdf_cif.loc[:, ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(residues_enumerated).astype('Int64')
        # pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(residues_enumerated).astype('Int64')
        expected_num_of_cols = 10
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # MAKE NEW COLUMN FOR ENUMERATED ATOMS ('C', 'CA', ETC), USING JSON->DICT `aa_atoms_enumerated` AND CAST TO INT:
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')
        expected_num_of_cols = 11
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # MAKE NEW COLUMN OF RESIDUE-ATOM PAIRS:
        # NEW COLUMN NAME = `aa_atom_tuple`. E.G. CONTAINS ('ASP':'C'), ('ASP':'CA'), ETC:
        pdf_cif[ColNames.AA_ATOM_PAIR.value] = list(zip(pdf_cif[CIF.S_mon_id.value], pdf_cif[CIF.A_label_atom_id.value]))

        # MAKE NEW COLUMN FOR ENUMERATED RESIDUE-ATOM PAIRS, VIA RESIDUE-ATOM PAIRS, THEN CAST TO INT:
        # NEW COLUMN NAME = `aa_atom_label_num`. E.G. CONTAINS 0, 386, 127, ETC.
        pdf_cif[ColNames.AA_ATOM_LABEL_NUM.value] = pdf_cif[ColNames.AA_ATOM_PAIR.value].map(atoms_enumerated).astype('Int64')
        expected_num_of_cols = 13
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # SUBTRACT EACH COORDINATE BY THE MEAN OF ALL 3 PER ATOM:
        pdf_cif.loc[:, ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                              CIF.A_Cartn_y.value,
                                                              CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif.loc[:, ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        expected_num_of_cols = 16
        assert len(pdf_cif.columns) == expected_num_of_cols, (f'Dataframe should have {expected_num_of_cols} columns. '
                                                              f'But this has {len(pdf_cif.columns)}')

        # WRITE OUT THE PARSED CIF TOKENS TO FLAT FILE (ssv BY DEFAULT):
        dh.write_tokenised_cif_to_flatfile(pdb_id, pdf_cif, dst_data_dir=relpath_to_dst_dir,
                                           flatfiles=flatfile_format_to_write)
        return pdf_cif


if __name__ == '__main__':

    # write_tokenised_cif_to_csv(pdb_ids='4itq')
    print(os.getcwd())
    # Being called from here, which is in the subdir `preprocessing_funcs` so paths must be specified
    parse_tokenise_cif_write_flatfile(pdb_ids='1OJ6', relpath_to_raw_cifs_dir='../diffusion/diff_data/cif/',
                                      relpath_to_dst_dir='../diffusion/diff_data/tokenised/')
    pass
