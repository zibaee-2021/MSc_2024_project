#!~/miniconda3/bin/python
from data_layer import data_handler
from src.preprocessing_funcs import tokeniser
from src.pLM import plm_embedder
from src.diffusion import pytorch_protfold_allatomclustrain_singlegpu


if __name__ == "__main__":

    # TOKENISE:
    data_handler.clear_diffdata_tokenised_dir()
    tokeniser.parse_tokenise_write_cifs_to_flatfile()

    # MAKE pLM EMBEDDINGS:
    data_handler.clear_diffdata_emb_dir()
    plm_embedder.generate_ankh_base_embeddings_from_tokenised_cifs()

    # TRAIN MODEL:
    pytorch_protfold_allatomclustrain_singlegpu.main()

