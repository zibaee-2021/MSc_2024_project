import json
from enum import Enum
import FASTA_reader as fasta_r


class Paths(Enum):
    per_residue_atoms_json = '../../data/residues_atoms/per_residue_atoms.json'


def _get_aa_to_atom_map() -> dict:
    relpath_json_f = Paths.per_residue_atoms_json.value
    try:
        with open(relpath_json_f, 'r') as json_f:
            aa_to_atoms_map = json.load(json_f)
    except FileNotFoundError:
        print(f'{relpath_json_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return aa_to_atoms_map


def translate_aa_to_atoms(uniprot_ids=None) -> dict:
    """
    Translate the given amino acid sequence to its corresponding atomic sequence using the PDB atom naming convention.
    :param uniprot_ids: Single uniprot id, list of uniprot_ids, or None by default. Most likely is a list of ids.
    :return: The atomic sequence, of amino acid sequence, mapped to its Uniprot id.
    """
    fasta_id_seqs = fasta_r.read_fasta_sequences(uniprot_ids=uniprot_ids)
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