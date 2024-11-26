from typing import Tuple
from enum import Enum
import torch
from data_layer import data_handler as dh
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModel


class Path(Enum):
    tokenised_dir = '../diffusion/diff_data/tokenised'
    local_emb_dir = '../diffusion/diff_data/emb'


class HFModelName(Enum):
    ANKH_BASE = 'ElnaggarLab/ankh-base'


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
        dh.save_torch_tensor(pt=aa_seq_embedding, dst_dir=f'{Path.local_emb_dir.value}/{pdbid}')
    return pdbid_raw_tok_emb


def generate_embeddings_from_aminoacid_sequence(tokeniser, eval_model, pdbid_chain: str, aa_sequence: str) -> dict:

    raw_tok_emb = {'raw_sequence': None, 'tokenised': None, 'embedding': None}
    pdbid_raw_tok_emb = {}

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
    pdbid_raw_tok_emb[pdbid_chain] = raw_tok_emb
    dh.save_torch_tensor(pt=aa_seq_embedding, dst_dir=Path.local_emb_dir.value, pdbid_chain=pdbid_chain)

    return pdbid_raw_tok_emb
