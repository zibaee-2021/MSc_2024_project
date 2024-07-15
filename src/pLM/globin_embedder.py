from typing import Tuple
from enum import Enum
import torch
from data_layer import data_handler as dh
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModel


class HFModelName(Enum):
    ANKH_BASE = 'ElnaggarLab/ankh-base'


def _load_tokeniser_and_eval_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    :param model_name:
    :return:
    """
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    eval_model = model.eval()
    return tokeniser, eval_model


def _generate_embeddings_from_fastas_of_pdbids(tokeniser, eval_model, pdbids_fastas):

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

    return pdbid_raw_tok_emb


def generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids():
    globins_pdb_fastas = dh.read_fastas_from_json_to_dict(filename='pdbids_sp_fastas_globins_5')
    tokeniser, eval_model = _load_tokeniser_and_eval_model(model_name=HFModelName.ANKH_BASE.value)
    globins_pdbid_raw_tok_emb = _generate_embeddings_from_fastas_of_pdbids(tokeniser=tokeniser,
                                                                           eval_model=eval_model,
                                                                           pdbids_fastas=globins_pdb_fastas)
    return globins_pdbid_raw_tok_emb


if __name__ == '__main__':

    globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids()
    pass

