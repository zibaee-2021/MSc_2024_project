import cif_parser as parser
from cif_parser import CIF
import pandas as pd
from enum import Enum
from data_layer import data_handler as dh


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'  # `A_label_comp_id` enumerated (the amino acid)
    ATOM_LABEL_NUM = 'atom_label_num'  # `A_label_atom_id` enumerated (the atom)
    BB_INDEX = 'bb_index'  # NOT CLEAR WHAT THIS IS .. BACKBONE ATOMS ?  ?  ?
    MEAN_COORDS = 'mean_xyz'  # mean of x y z coordinates for each atom
    MEAN_CORR_X = 'mean_corrected_x'  # x coordinates for each atom subtracted by the mean of xyz coordinates
    MEAN_CORR_Y = 'mean_corrected_y'  # (as above) but for y coordinates
    MEAN_CORR_Z = 'mean_corrected_z'  # (as above) but for z coordinates


def _write_to_csv(pdb_id: str, pdf: pd.DataFrame):
    # pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.csv', index=False, na_rep='null')
    # pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.csv', index=False)
    # pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.ssv', sep=' ', index=False, na_rep='null')
    pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.ssv', sep=' ', index=False)
    # pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.tsv', sep='\t', index=False, na_rep='null')
    # pdf.to_csv(path_or_buf=f'../data/tokenised/{pdb_id}.tsv', sep='\t', index=False)
    pdf_easy_read = pdf.rename(columns={CIF.S_seq_id.value: 'SEQ_ID',
                                        CIF.S_mon_id.value: 'RESIDUES',
                                        CIF.A_id.value: 'ATOM_ID',
                                        CIF.A_label_atom_id.value: 'ATOMS',
                                        ColNames.MEAN_CORR_X.value: 'X',
                                        ColNames.MEAN_CORR_Y.value: 'Y',
                                        ColNames.MEAN_CORR_Z.value: 'Z'})
    # pdf_easy_read.to_csv(path_or_buf=f'../data/tokenised/easyRead_{pdb_id}.tsv', sep='\t', index=False, na_rep='null')


def write_tokenised_cif_to_csv(pdb_ids=None) -> None:
    """
    Tokenise the mmCIF files for the specified proteins by PDB entry/entries (which is a unique identifier) and write
    to csv (and/or tsv and/or ssv) files at `../data/tokenised/`.
    :param pdb_ids: PDB identifier(s) for protein(s) to tokenise.
    """
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]

    for pdb_id in pdb_ids:
        pdf_cif = parser.parse_cif(pdb_id=pdb_id, local_cif_file=f'../data/cifs/{pdb_id}.cif')

        atoms_enumerated, aas_enumerated = dh.read_enumeration_mappings()

        # Amino acid labels enumerated
        pdf_cif[ColNames.AA_LABEL_NUM.value] = pdf_cif[CIF.S_mon_id.value].map(aas_enumerated).astype('Int64')

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
        _write_to_csv(pdb_id, pdf_cif)


if __name__ == '__main__':

    write_tokenised_cif_to_csv(pdb_ids='4itq')
