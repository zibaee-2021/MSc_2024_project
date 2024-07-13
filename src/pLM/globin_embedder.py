from data_layer import data_handler as dh
import HF_tokenisers_pLMs as hf


def generate_embeddings_from_fastas_of_pdbids(tokeniser, eval_model, pdbids_fastas):

    raw_tok_emb = {'raw_sequence': None, 'tokenised': None, 'embedding': None}
    pdbid_raw_tok_emb = {'pdb_id': raw_tok_emb}

    for pdbid in pdbids_fastas:
        aa_sequence = pdbids_fastas[pdbid]['fasta']
        raw_tok_emb['raw_sequence'] = aa_sequence

        tokenised_aa_sequence = tokeniser(aa_sequence, return_tensors='pt')
        raw_tok_emb['tokenised'] = tokenised_aa_sequence
        seq_len = len(aa_sequence)

        output = eval_model(**tokenised_aa_sequence)
        aa_sequence_embedding = output[0]
        raw_tok_emb['embedding'] = aa_sequence_embedding

        pdbid_raw_tok_emb[pdbid] = raw_tok_emb

    return pdbid_raw_tok_emb


def generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids():
    globins_pdb_fastas = dh.read_fastas_from_json_to_dict(filename='pdbids_sp_fastas_globins_5')
    globins_pdbid_raw_tok_emb = generate_embeddings_from_fastas_of_pdbids(tokeniser=hf.ankh_base_tokeniser(),
                                                                          eval_model=hf.ankh_base_model_for_eval(),
                                                                          pdbids_fastas=globins_pdb_fastas)
    return globins_pdbid_raw_tok_emb


if __name__ == '__main__':

    globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids()
    pass

