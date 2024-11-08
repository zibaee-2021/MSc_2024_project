import json
import fasta_reader as reader
from data_layer import data_handler as dh


def _get_aa_to_atom_map() -> dict:
    aa_atoms_path = '../../data/atoms/per_residue_atoms.json'
    try:
        with open(aa_atoms_path, 'r') as json_f:
            aa_to_atoms_map = json.load(json_f)
    except FileNotFoundError:
        print(f'{aa_atoms_path} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return aa_to_atoms_map


def translate_aa_to_atoms(uniprot_ids=None) -> dict:
    """
    Translate the given amino acid sequence to its corresponding atomic sequence using the PDB atom naming convention.
    :param uniprot_ids: Single uniprot id, list of uniprot_ids, or None by default. Most likely is a list of ids.
    :return: The atomic sequence, of amino acid sequence, mapped to its Uniprot id.
    """
    fasta_id_seqs = reader.read_fasta_sequences(uniprot_ids=uniprot_ids)
    aa_to_atoms_map = _get_aa_to_atom_map()
    id_to_atomic_sequence = dict()

    for uniprot_id, fastaa in fasta_id_seqs.items():
        atomic_sequence = []
        for aa in fastaa:
            if aa in aa_to_atoms_map:
                atomic_sequence.extend(aa_to_atoms_map[aa])
            else:
                raise ValueError(f'{aa} is not one of the 20 amino acids (using 1-letter format)')
            atomic_sequence = ''.join(atomic_sequence)
        id_to_atomic_sequence[uniprot_id] = atomic_sequence
    return id_to_atomic_sequence


if __name__ == '__main__':

    translate_aa_to_atoms(uniprot_ids=None)