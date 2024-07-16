from data_layer import data_handler as dh
import plm_embedder as pe


def generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids():
    globins_pdb_fastas = dh.read_nonnull_fastas_from_json_to_dict(filename='pdbids_sp_fastas_globins_5')
    tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name=pe.HFModelName.ANKH_BASE.value)
    globins_pdbid_raw_tok_emb = pe.generate_embeddings_from_fastas_of_pdbids(tokeniser=tokeniser,
                                                                             eval_model=eval_model,
                                                                             pdbids_fastas=globins_pdb_fastas)
    return globins_pdbid_raw_tok_emb


if __name__ == '__main__':

    globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids()
    pass

