#!~/miniconda3/bin/python
"""
Script to generate protein language model embeddings for given amino acid sequences, using pretrained
sequence-to-sequence model, with accompanying tokeniser, supplied via HuggingFace transformers library.
Embeddings are written to file (`.pt`) files.
"""
import os
from typing import Tuple
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModel
from data_layer import data_handler as dh
from src.preprocessing_funcs import tokeniser as tk


def _generate_embeddings_from_aminoacid_sequence(hf_tokeniser, hf_eval_model, pdbid_chain: str, aa_seq: str) -> dict:
    """
    Generate protein language embeddings for given amino acid sequence, using the given HuggingFace tokeniser and
    pretrained sequence-to-sequence protein language model, in eval mode. Write embedding to file, (`.pt`).
    :param hf_tokeniser: Specific HuggingFace tokeniser for `hf_eval_model` protein language model.
    :param hf_eval_model: Specific HuggingFace pretrained protein language model, in eval mode.
    :param pdbid_chain: Identifier of protein sequence language model embedding is being created for, e.g. '1ECA_A'
    :param aa_seq: Amino acid sequence (1-letter format).
    """
    inputs = hf_tokeniser(aa_seq, return_tensors='pt')
    decoder_input_ids = inputs['input_ids']
    # REMOVE EOS TOKEN:
    decoder_input_ids = decoder_input_ids[:, :-1]

    # with torch.no_grad():
    embedding = hf_eval_model(input_ids=decoder_input_ids,
                              decoder_input_ids=decoder_input_ids)
    # embedding = embedding.last_hidden_state
    # aa_sequence_embedding = embedding[0]
    aa_seq_embedding = embedding.encoder_last_hidden_state
    aa_seq_embedding = aa_seq_embedding.detach()
    dh.save_torch_tensor_to_pt(pt_tensor_to_save=aa_seq_embedding,
                               dst_dir='../diffusion/diff_data/emb',
                               pdbid_chain=pdbid_chain)


def _load_tokeniser_and_eval_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Instantiate tokeniser and instantiate pre-trained sequence-to-sequence language model, in evaluation mode.
    :param model_name: HuggingFace name of tokeniser and pre-trained model to instantiate from HuggingFace repositories.
    :return: Specified tokeniser and model in eval mode.
    """
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    eval_model = model.eval()
    return tokeniser, eval_model


def _get_aa_sequence_from_ssv(pdbid_chain: str) -> str:
    """
    Get amino acid sequence (1-letter format) for given `pdbid_chain`, reading it directly from its ssv file in
    `/diffusion/diff_data/tokenised` directory.
    :param pdbid_chain: Identifier of which amino acid sequence to return. E.g. '1ECA_A'.
    :return: Amino acid sequence (1-letter format) for the specified `pdbid_chain`.
    """
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


def generate_ankh_base_embeddings_from_tokenised_cifs(pdbid_chain: str):
    aa_sequence = _get_aa_sequence_from_ssv(pdbid_chain)
    hf_tokeniser, hf_eval_model = _load_tokeniser_and_eval_model(model_name='ElnaggarLab/ankh-base')
    _generate_embeddings_from_aminoacid_sequence(hf_tokeniser=hf_tokeniser,
                                                 hf_eval_model=hf_eval_model,
                                                 pdbid_chain=pdbid_chain,
                                                 aa_seq=aa_sequence)


if __name__ == '__main__':

    _abs_path = os.path.dirname(os.path.abspath(__file__))
    # lst_file = 'pdbchains_9.lst'
    lst_file = 'pdbchains_565.lst'
    _targetfile_lst_path = f'../diffusion/diff_data/PDBid_list/{lst_file}'
    abspath_lst_file = os.path.normpath(os.path.join(_abs_path, _targetfile_lst_path))

    assert os.path.exists(abspath_lst_file), f'{abspath_lst_file} cannot be found.'

    _pdbid_chains = dh.read_pdb_lst_file(relpath_pdblst=abspath_lst_file)

    for _pdbid_chain in _pdbid_chains:
        generate_ankh_base_embeddings_from_tokenised_cifs(_pdbid_chain)


