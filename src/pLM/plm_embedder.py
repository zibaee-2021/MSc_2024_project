import os
from typing import Tuple
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModel
from data_layer import data_handler as dh
from src.preprocessing_funcs import tokeniser as tk


def load_tokeniser_and_eval_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    :param model_name:
    :return:
    """
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    eval_model = model.eval()
    return tokeniser, eval_model


def generate_embeddings_from_fastas_of_pdbids(tokeniser, eval_model, pdbids_fastas: dict) -> dict:

    raw_tok_emb = {'raw_sequence': None, 'tokenised': None, 'embedding': None}
    pdbid_raw_tok_emb = {}

    for pdbid in pdbids_fastas:
        aa_sequence = pdbids_fastas[pdbid]['fasta']
        raw_tok_emb['raw_sequence'] = aa_sequence
        raw_tok_emb['seq_len'] = len(aa_sequence)

        inputs = tokeniser(aa_sequence, return_tensors='pt')
        decoder_input_ids = inputs['input_ids']
        raw_tok_emb['tokenised'] = decoder_input_ids

        # with torch.no_grad():
        embedding = eval_model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)

        # embedding = embedding.last_hidden_state
        # aa_sequence_embedding = embedding[0]
        aa_seq_embedding = embedding.encoder_last_hidden_state
        raw_tok_emb['embedding'] = aa_seq_embedding
        pdbid_raw_tok_emb[pdbid] = raw_tok_emb
        dh.save_torch_tensor(pt=aa_seq_embedding, dst_dir=f'../diffusion/diff_data/emb/{pdbid}', pdbid_chain=pdbid)
    return pdbid_raw_tok_emb


def generate_embeddings_from_aminoacid_sequence(tokeniser, eval_model, pdbid_chain: str, aa_sequence: str) -> dict:

    raw_tok_emb = {'raw_sequence': None, 'tokenised': None, 'embedding': None}
    pdbid_raw_tok_emb = {}

    raw_tok_emb['raw_sequence'] = aa_sequence
    raw_tok_emb['seq_len'] = len(aa_sequence)

    inputs = tokeniser(aa_sequence, return_tensors='pt')
    decoder_input_ids = inputs['input_ids']
    # REMOVE EOS TOKEN:
    decoder_input_ids = decoder_input_ids[:, :-1]
    raw_tok_emb['tokenised'] = decoder_input_ids

    # with torch.no_grad():
    embedding = eval_model(input_ids=decoder_input_ids, decoder_input_ids=decoder_input_ids)
    # embedding = embedding.last_hidden_state
    # aa_sequence_embedding = embedding[0]
    aa_seq_embedding = embedding.encoder_last_hidden_state
    aa_seq_embedding = aa_seq_embedding.detach()
    raw_tok_emb['embedding'] = aa_seq_embedding
    pdbid_raw_tok_emb[pdbid_chain] = raw_tok_emb
    dh.save_torch_tensor(pt=aa_seq_embedding, dst_dir='../diffusion/diff_data/emb', pdbid_chain=pdbid_chain)

    return pdbid_raw_tok_emb


def _get_aa_sequence_from_ssv(pdbid_chain: str) -> str:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_tokenised_ssv = f'../diffusion/diff_data/tokenised/{pdbid_chain}.ssv'
    abspath_tokenised_ssv = os.path.normpath(os.path.join(abs_path, path_tokenised_ssv))
    pdf = dh.read_tokenised_cif_chain_ssv_to_pdf(abspath_tokenised_ssv)
    tk.get_nums_of_missing_data(pdf)
    aa_pos_seq_pdf = pdf[['S_seq_id', 'S_mon_id']]
    aa_pos_seq_pdf = aa_pos_seq_pdf.drop_duplicates(subset='S_seq_id', keep='first')
    aa_seq = aa_pos_seq_pdf['S_mon_id']
    aa_seq = aa_seq.tolist()
    aa_3to1 = dh.read_aa_3to1_yaml()
    aa_sequence = ''.join([aa_3to1[aa] for aa in aa_seq])
    len_aa_seq = len(aa_sequence)
    return aa_sequence


def generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(pdbid_chain: str):
    aa_sequence = _get_aa_sequence_from_ssv(pdbid_chain)
    len_aa_seq = len(aa_sequence)
    # aa_sequence = 'AMG'  # just testing
    tokeniser, eval_model = load_tokeniser_and_eval_model(model_name='ElnaggarLab/ankh-base')
    pdbid_raw_tok_emb = generate_embeddings_from_aminoacid_sequence(tokeniser=tokeniser,
                                                                               eval_model=eval_model,
                                                                               pdbid_chain=pdbid_chain,
                                                                               aa_sequence=aa_sequence)
    return pdbid_raw_tok_emb


if __name__ == '__main__':

    abs_path = os.path.dirname(os.path.abspath(__file__))
    # lst_file = 'pdbchains_9.lst'
    lst_file = 'pdbchains_565.lst'
    _targetfile_lst_path = f'../diffusion/diff_data/PDBid_list/{lst_file}'
    abspath_lst_file = os.path.normpath(os.path.join(abs_path, _targetfile_lst_path))

    assert os.path.exists(abspath_lst_file), f'{abspath_lst_file} cannot be found. Btw, cwd={os.getcwd()}'

    _pdbid_chains = dh.read_pdb_lst_file(relpath_pdblst=abspath_lst_file)
    for _pdbid_chain in _pdbid_chains:
        globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(_pdbid_chain)


