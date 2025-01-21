import os
import json
from src.preprocessing_funcs import FASTA_reader as fasta_r
from data_layer import data_handler as dh


def _get_aa_to_atom_map() -> dict:
    relpath_json_f = '../../data/residues_atoms/per_residue_atoms.json'
    try:
        with open(relpath_json_f, 'r') as json_f:
            aa_to_atoms_map = json.load(json_f)
    except FileNotFoundError:
        print(f'{relpath_json_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return aa_to_atoms_map


def translate_aa_to_atoms(uniprot_ids=None, pdbid_chains=None) -> dict:

    """
    Translate the given amino acid sequence to its corresponding atomic sequence using the PDB atom naming convention.
    :param uniprot_ids: Single uniprot id, list of uniprot_ids, or None by default. Most likely is a list of ids.
    :param pdbid_chains: One or more PDBids to read from tokenised folder. Can include chain suffix or not.
    :return: The atomic sequence, of amino acid sequence, mapped to its Uniprot id.
    """
    aa_to_atoms_map = _get_aa_to_atom_map()
    id_to_atomic_sequence = dict()
    atomic_sequence = []

    if uniprot_ids:
        fasta_id_seqs = fasta_r.read_fasta_sequences(uniprot_ids=uniprot_ids)

        for uniprot_id, fasta_prot in fasta_id_seqs.items():
            for aa in fasta_prot:
                if aa in aa_to_atoms_map:
                    atomic_sequence.extend(aa_to_atoms_map[aa])
                else:
                    raise ValueError(f'{aa} is not one of the 20 amino acids (using 1-letter format)')
                atomic_sequence = ''.join(atomic_sequence)
            id_to_atomic_sequence[uniprot_id] = atomic_sequence
    else:
        assert pdbid_chains, print('Expecting pdbid_chain to be provided. Something has gone wrong.')

        if isinstance(pdbid_chains, str):
            pdbid_chains = [pdbid_chains]

        abs_path = os.path.dirname(os.path.abspath(__file__))
        for pdbid_chain in pdbid_chains:
            path_tokenised_dir = f'../diffusion/diff_data/tokenised'
            abspath_tknsd_dir = os.path.normpath(os.path.join(abs_path, path_tokenised_dir))
            pdf = dh.read_tokenised_pdbid_chain_ssv_to_pdf(path_tokensd_dir=abspath_tknsd_dir, pdbid_chain=pdbid_chain)
            aa_position_sequence = pdf[['S_seq_id', 'S_mon_id']]
            aa_position_sequence = aa_position_sequence.drop_duplicates(subset='S_seq_id', keep='first')
            aa_sequence = ''.join(aa_position_sequence['S_mon_id'])
            for aa in aa_sequence:
                if aa in aa_to_atoms_map:
                    atomic_sequence.extend(aa_to_atoms_map[aa])
                else:
                    raise ValueError(f'{aa} is not one of the 20 amino acids (using 1-letter format)')
                atomic_sequence = ''.join(atomic_sequence)
            id_to_atomic_sequence[pdbid_chain] = atomic_sequence

    return id_to_atomic_sequence


if __name__ == '__main__':

    translate_aa_to_atoms(uniprot_ids=None)