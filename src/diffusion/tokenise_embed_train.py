#!~/miniconda3/bin/python
import os
import sys
from data_layer import data_handler
from src.preprocessing_funcs import tokeniser
from src.pLM import plm_embedder
from src.diffusion import pytorch_protfold_allatomclustrain_singlegpu


if __name__ == "__main__":

    # TOKENISE:
    data_handler.clear_diffdata_mmcif_dir()
    data_handler.clear_diffdata_tokenised_dir()

    if len(sys.argv) > 1:
        relpath_pdblst = f'../diffusion/diff_data//PDBid_list/{sys.argv[1]}.lst'
        print(f'Argument received --> will use this list of PDBids: {relpath_pdblst}')

        # 5. PARSE AND TOKENISE MMCIF FILES AND WRITE SSV FILES TO `DIFFUSION/DIFF_DATA/TOKENISED` DIRECTORY:
        tokeniser.parse_tokenise_write_cifs_to_flatfile(relpath_pdblst=relpath_pdblst)
    else:
        print('No PDBid list filename argument provided.')
        # 5. PARSE AND TOKENISE MMCIF FILES AND WRITE SSV FILES TO `DIFFUSION/DIFF_DATA/TOKENISED` DIRECTORY:
        tokeniser.parse_tokenise_write_cifs_to_flatfile()

    # MAKE pLM EMBEDDINGS:
    data_handler.clear_diffdata_emb_dir()
    plm_embedder.generate_plm_embeddings_from_tokenised_cifs_using_pretrained_models()

    # TRAIN MODEL:
    pytorch_protfold_allatomclustrain_singlegpu.main()

