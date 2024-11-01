"""
TOKENISER.PY
    - CALLS CIF PARSER TO READ IN AND PARSE THE CIF FILE TO EXTRACT THE FOLLOWING 14 FIELDS. WRITE TO .ssv FLATFILE.
    - ENUMERATES ATOMS AND AMINO ACID RESIDUES.
    - SUBTRACTS COORDINATES BY THEIR MEAN COORDINATE VALUES PER ATOM.
----------------------------------------------------------------------------------------------------------------------
These 14 fields are used and end up in a 14-column dataframe. A description of what they are all used for is given here
and below (I am happy to repeat myself in an effort to reduce the chance of mistakes due to confusing names).

atom_site:
    group_PDB,          # 'ATOM' or 'HETATM'    - FILTER ON THIS, THEN REMOVE IT.
    auth_seq_id,        # RESIDUE POSITION      - USED TO JOIN WITH S_pdb_seq_num, THEN REMOVE IT.
    label_comp_id,      # RESIDUE (3-LETTER)    - USED TO SANITY-CHECK WITH S_mon_id, THEN REMOVE IT.
    id,                 # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
    label_atom_id,      # ATOM                  - KEEP IN DF.
    label_asym_id,      # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
    Cartn_x,            # ATOM COORDS          - X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_y,            # ATOM COORDS           - Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    Cartn_z,            # ATOM COORDS           - Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN)
    occupancy           # OCCUPANCY             - FILTER ON THIS, THEN REMOVE IT.

_pdbx_poly_seq_scheme:
    seq_id,             # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
    mon_id,             # RESIDUE (3-LETTER)    - USE FOR SANITY-CHECK AGAINST A_label_comp_id, KEEP IN DF.
    pdb_seq_num,        # RESIDUE POSITION      - JOIN TO A_auth_seq_id, THEN REMOVE IT.
    asym_id,            # CHAIN                 - JOIN ON THIS, SORT ON THIS, THEN REMOVE IT.
"""


import os
import pandas as pd
from src.preprocessing_funcs import cif_parser as parser
from data_layer import data_handler as dh
from src.enums import ColNames, CIF


def parse_tokenise_cif_write_flatfile(pdb_ids=None, flatfileformat_to_write: str = 'ssv',
                                      path_to_raw_cifs_dir='data/cif',
                                      dst_path_for_tokenised='data/tokenised') -> pd.DataFrame:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `data/tokenised/`.
    Specifically: enumerate the atoms and residues. Correct x, y, z coordinates by the mean of all 3 per row.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    :param flatfileformat_to_write: Write to ssv, csv or tsv. Use ssv by default.
    :param path_to_raw_cifs_dir: Path to source dir of the raw cif files to be parsed and tokenised.
    Use `diffusion/data/cif` subdir by default. Expectation is that this is mostly called from diffusion dir.
    :param dst_path_for_tokenised: Path to destination dir for the parsed and tokenised cif as a flat file.
    Use `diffusion/data/tokenised` subdir by default. Expectation is that this is mostly called from `diffusion` dir.
    If the caller passes in empty string '', it will be interpreted as instruction to write to `diffusion/data/tokenised`.
    :return: Parsed and tokenised cif file in dataframe, which is also written to ssv in `data/tokenised`.
    Dataframe with 13 Columns: 'A_label_asym_id', 'S_seq_id', 'A_id', 'A_label_atom_id', 'A_Cartn_x', 'A_Cartn_y',
    'A_Cartn_z', 'aa_label_num', 'atom_label_num', 'mean_xyz', 'mean_corrected_x', 'mean_corrected_y',
    'mean_corrected_z']. NB: 'A_Cartn_x', 'A_Cartn_y', 'A_Cartn_z' and 'mean_xyz' are no longer needed but I'm
    leaving them for sanity-checks (by eye).
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        path_to_raw_cifs_dir = path_to_raw_cifs_dir.removesuffix('.cif').removeprefix('/')
        cif = f'{path_to_raw_cifs_dir}{pdb_id}.cif'
        assert os.path.exists(cif)

        # PARSE mmCIF TO EXTRACT 14 FIELDS, TO FILTER, IMPUTE, SORT AND JOIN ON, RETURNING AN 8-COLUMN DATAFRAME:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, path_to_raw_cif=cif)
        assert len(pdf_cif.columns) == 8, f'Dataframe should have 8 columns. But this has {len(pdf_cif.columns)}'

        # READ ENUMERATION MAPPINGS TO DICTS TO USE FOR CONVERTING RESIDUES AND ATOMS TO NUMBERS:
        atoms_enumerated, aas_enumerated, fasta_aas_enumerated = dh.read_enumeration_mappings()

        # ENUMERATE BY MAPPING RESIDUES, USING `aa_atoms_enumerated` JSON->DICT AND CAST TO INT:
        pdf_cif.loc[:, ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')
        assert len(pdf_cif.columns) == 9, f'Dataframe should have 9 columns. But this has {len(pdf_cif.columns)}'

        # ENUMERATE BY MAPPING ATOM ('C', 'CA', ETC), USING JSON->DICT `aa_atoms_enumerated` AND CAST TO INT:
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')
        assert len(pdf_cif.columns) == 10, f'Dataframe should have 10 columns. But this has {len(pdf_cif.columns)}'

        # SUBTRACT EACH COORDINATE BY THE MEAN OF ALL 3 PER ATOM:
        pdf_cif.loc[:, ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                              CIF.A_Cartn_y.value,
                                                              CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif.loc[:, ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif.loc[:, ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]

        # REMOVE ORIGINAL NON-ENUMERATED RESIDUE COLUMN:
        pdf_cif = pdf_cif.drop(columns=[CIF.S_mon_id.value])
        assert len(pdf_cif.columns) == 13, f'Dataframe should have 13 columns. But this has {len(pdf_cif.columns)}'

        # WRITE OUT THE PARSED CIF TOKENS TO FLAT FILE (.ssv BY DEFAULT):
        dh.write_tokenised_cif_to_flatfile(pdb_id, pdf_cif, dst_data_dir=dst_path_for_tokenised,
                                           flatfiles=flatfileformat_to_write)

        return pdf_cif


if __name__ == '__main__':

    # write_tokenised_cif_to_csv(pdb_ids='4itq')
    print(os.getcwd())
    # Being called from here, which is in the subdir `preprocessing_funcs` so paths must be specified
    parse_tokenise_cif_write_flatfile(pdb_ids='1OJ6', path_to_raw_cifs_dir='../diffusion/data/cif/',
                                      dst_path_for_tokenised='../diffusion/data/tokenised/')
    pass
