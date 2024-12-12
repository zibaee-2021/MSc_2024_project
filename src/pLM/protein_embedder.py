import os
# from enum import Enum
from data_layer import data_handler as dh
import plm_embedder as pe
# from plm_embedder import Path
# from src.enums import CIF
from src.preprocessing_funcs import tokeniser as tk


# class Filename(Enum):
#     pdbid_fasta_globs = 'PDBid_sp_FASTA_Globins'


def generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids():
    # globins_pdb_fastas = dh.read_nonnull_fastas_from_json_to_dict(fname=Filename.pdbid_fasta_globs.value)
    globins_pdb_fastas = dh.read_nonnull_fastas_from_json_to_dict(fname='PDBid_sp_FASTA_Globins')
    # tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name=pe.HFModelName.ANKH_BASE.value)
    tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name='ElnaggarLab/ankh-base')
    globins_pdbid_raw_tok_emb = pe.generate_embeddings_from_fastas_of_pdbids(tokeniser=tokeniser,
                                                                             eval_model=eval_model,
                                                                             pdbids_fastas=globins_pdb_fastas)
    return globins_pdbid_raw_tok_emb


def _get_aa_sequence_from_ssv(pdbid_chain: str) -> str:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_tokenised_ssv = f'../diffusion/diff_data/tokenised/{pdbid_chain}.ssv' 
    abspath_tokenised_ssv = os.path.normpath(os.path.join(abs_path, path_tokenised_ssv))
    pdf = dh.read_tokenised_cif_chain_ssv_to_pdf(abspath_tokenised_ssv)
    tk.nums_of_missing_data(pdf)
    # aa_pos_seq_pdf = pdf[[CIF.S_seq_id.value, CIF.S_mon_id.value]]
    aa_pos_seq_pdf = pdf[['S_seq_id', 'S_mon_id']]
    # aa_pos_seq_pdf = aa_pos_seq_pdf.drop_duplicates(subset=CIF.S_seq_id.value, keep='first')
    aa_pos_seq_pdf = aa_pos_seq_pdf.drop_duplicates(subset='S_seq_id', keep='first')
    # aa_seq = aa_pos_seq_pdf[CIF.S_mon_id.value]
    aa_seq = aa_pos_seq_pdf['S_mon_id']
    aa_seq = aa_seq.tolist()
    aa_3to1 = dh.read_aa_3to1_yaml()
    aa_sequence = ''.join([aa_3to1[aa] for aa in aa_seq])
    len_aa_seq = len(aa_sequence)
    return aa_sequence


def generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(pdbid_chain: str):
    aa_sequence = _get_aa_sequence_from_ssv(pdbid_chain)
    len_aa_seq = len(aa_sequence)
    # aa_sequence = 'AMG'
    # tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name=pe.HFModelName.ANKH_BASE.value)
    tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name='ElnaggarLab/ankh-base')
    globins_pdbid_raw_tok_emb = pe.generate_embeddings_from_aminoacid_sequence(tokeniser=tokeniser,
                                                                               eval_model=eval_model,
                                                                               pdbid_chain=pdbid_chain,
                                                                               aa_sequence=aa_sequence)
    return globins_pdbid_raw_tok_emb


if __name__ == '__main__':

    # globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids()
    # globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(pdbid_chain='1ECA_A')

    # _pdbid_chains = dh.read_pdb_lst_from_src_diff_dir(relpath_pdblst=f'{dh.Path.rp_diffdata_pdbid_lst_dir.value}/'
                                                                    #  f'pdbchains_9{dh.FileExt.dot_lst.value}')
    # _targetfile_lst_path = Path.rp_diffdata_9_PDBids_lst.value
    # lst_file = 'pdbchains_9.lst'
    abs_path = os.path.dirname(os.path.abspath(__file__))
    lst_file = 'pdbchains_565.lst'
    _targetfile_lst_path = f'../diffusion/diff_data/PDBid_list/{lst_file}'
    abspath_lst_file = os.path.normpath(os.path.join(abs_path, _targetfile_lst_path))

    assert os.path.exists(abspath_lst_file), f'{abspath_lst_file} cannot be found. Btw, cwd={os.getcwd()}'

    _pdbid_chains = dh.read_pdb_lst_file(relpath_pdblst=abspath_lst_file)
    for _pdbid_chain in _pdbid_chains:
        globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(_pdbid_chain)
        pass
    pass

