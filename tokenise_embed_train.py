#!/usr/bin/env python3
import os
from data_layer import data_handler
from src.preprocessing_funcs import tokeniser
from src.pLM import plm_embedder
from src.diffusion import pytorch_protfold_allatomclustrain_singlegpu


if __name__ == "__main__":

    # TOKENISE:
    data_handler.clear_diffdata_tokenised_dir()
    _, lst_file = tokeniser.parse_tokenise_write_cifs_to_flatfile()

    # MAKE pLM EMBEDDINGS:
    data_handler.clear_diffdata_emb_dir()
    plm_embedder.generate_ankh_base_embeddings_from_tokenised_cifs()

    # TRAIN MODEL:
    _abs_path = os.path.dirname(os.path.abspath(__file__))
    _targetfile_lst_path = os.path.normpath(os.path.join(_abs_path, f'diff_data/PDBid_list/{lst_file}'))
    assert os.path.exists(_targetfile_lst_path), f'{_targetfile_lst_path} cannot be found. Forget about it.'
    pytorch_protfold_allatomclustrain_singlegpu.main(_targetfile_lst_path)

