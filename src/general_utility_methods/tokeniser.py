import os

import pandas as pd
from src.general_utility_methods import cif_parser as parser
from src.general_utility_methods.cif_parser import CIF
from data_layer import data_handler as dh
from enums.colnames import ColNames


def parse_tokenise_cif_write_to_flatfile_to_pdf(pdb_ids=None, use_local_data_subdir=False, flatfile: str = 'ssv') -> pd.DataFrame:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `../data/tokenised/`.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    :param use_local_data_subdir: True to write the tokenised PDB values to `data` subdir in cwd, otherwise write to the
    larger, general-use, `data` dir that still at top-level of project structure. False by default.
    :param flatfile: Write to ssv, csv or tsv. Use ssv by default.
    :return parsed and tokenised cif file in dataframe, which is also written to ssv in `data/tokenised`.
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        # os.getcwd() gives 'diffSock/src/diffusion'
        path_to_cif_pdb_ids = f'{os.getcwd()}/data/cif/'
        assert os.path.exists(path_to_cif_pdb_ids)
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, local_cif_file=f'data/cif/{pdb_id}.cif')

        atoms_enumerated, aas_enumerated, fasta_aas_enumerated = dh.read_enumeration_mappings()

        # # Amino acid labels enumerated
        # pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')

        # FASTA Amino acid labels enumerated
        pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(fasta_aas_enumerated).astype('Int64')

        # Atom labels enumerated
        pdf_cif[ColNames.ATOM_LABEL_NUM.value] = pdf_cif[CIF.A_label_atom_id.value].map(atoms_enumerated).astype('Int64')

        # Atomic xyz coordinates
        pdf_cif[ColNames.MEAN_COORDS.value] = pdf_cif[[CIF.A_Cartn_x.value,
                                                       CIF.A_Cartn_y.value,
                                                       CIF.A_Cartn_z.value]].mean(axis=1)
        pdf_cif[ColNames.MEAN_CORR_X.value] = pdf_cif[CIF.A_Cartn_x.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Y.value] = pdf_cif[CIF.A_Cartn_y.value] - pdf_cif[ColNames.MEAN_COORDS.value]
        pdf_cif[ColNames.MEAN_CORR_Z.value] = pdf_cif[CIF.A_Cartn_z.value] - pdf_cif[ColNames.MEAN_COORDS.value]

        # ONLY KEEP THESE COLUMNS:
        pdf_cif = pdf_cif[[CIF.S_seq_id.value,
                           CIF.S_mon_id.value,
                           CIF.A_id.value,
                           CIF.A_label_atom_id.value,
                           ColNames.MEAN_CORR_X.value,
                           ColNames.MEAN_CORR_Y.value,
                           ColNames.MEAN_CORR_Z.value]]
        if flatfile == 'ssv':
            dh.write_tokenised_cif_to_flatfile(pdb_id, pdf_cif, use_local_data_subdir=use_local_data_subdir,
                                               flatfiles='ssv')
        return pdf_cif


if __name__ == '__main__':

    # write_tokenised_cif_to_csv(pdb_ids='4itq')
    parse_tokenise_cif_write_to_flatfile_to_pdf(pdb_ids='1oj6')
