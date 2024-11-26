from enum import Enum
from data_layer import data_handler as dh
import plm_embedder as pe
from plm_embedder import Path
from src.enums import CIF


class Filename(Enum):
    pdbid_fasta_globs = 'PDBid_sp_FASTA_Globins'


def generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids():
    globins_pdb_fastas = dh.read_nonnull_fastas_from_json_to_dict(fname=Filename.pdbid_fasta_globs.value)
    tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name=pe.HFModelName.ANKH_BASE.value)
    globins_pdbid_raw_tok_emb = pe.generate_embeddings_from_fastas_of_pdbids(tokeniser=tokeniser,
                                                                             eval_model=eval_model,
                                                                             pdbids_fastas=globins_pdb_fastas)
    return globins_pdbid_raw_tok_emb


def generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(pdbid_chain: str):

    pdf = dh.read_tokenised_cif_chain_ssv_to_pdf(pdbid_chain=pdbid_chain, relpath_tokensd_dir=Path.tokenised_dir.value)
    aa_position_sequence = pdf[[CIF.S_seq_id.value, CIF.S_mon_id.value]]
    aa_position_sequence = aa_position_sequence.drop_duplicates(subset=CIF.S_seq_id.value, keep='first')
    aa_sequence = ''.join(aa_position_sequence[CIF.S_mon_id.value])
    tokeniser, eval_model = pe.load_tokeniser_and_eval_model(model_name=pe.HFModelName.ANKH_BASE.value)
    globins_pdbid_raw_tok_emb = pe.generate_embeddings_from_aminoacid_sequence(tokeniser=tokeniser,
                                                                               eval_model=eval_model,
                                                                               pdb_id_chain=pdbid_chain,
                                                                               aa_sequence=aa_sequence)
    return globins_pdbid_raw_tok_emb


if __name__ == '__main__':

    # globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_of_5_globins_from_fastas_of_pdbids()
    globins_pdbid_raw_tok_emb_ = generate_ankh_base_embeddings_from_seq_id_of_tokenised_cifs(pdbid_chain='1ECA_A')
    pass

