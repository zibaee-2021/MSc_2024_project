import json
import fasta_reader as reader


def _get_aa_to_atom_map() -> dict:
    aa_atoms_path = '../data/jsons/aa_atoms.json'
    try:
        with open(aa_atoms_path, 'r') as json_f:
            aa_to_atoms_map = json.load(json_f)
    except FileNotFoundError:
        print(f'{aa_atoms_path} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return aa_to_atoms_map


def manually_write_aa_atoms_to_data_dir(path: str) -> None:
    """
    This function only needs to be done once.
    :param path:
    :return:
    """
    aa_atoms = {
        'A': ['N', 'CA', 'C', 'O', 'CB'],
        'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
        'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
        'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
        'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'G': ['N', 'CA', 'C', 'O'],
        'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
        'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
        'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
        'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
        'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
        'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
        'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
        'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
        'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
        'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
        'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
        'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
    }
    with open(path, 'w') as json_f:
        json.dump(aa_atoms, json_f, indent=4)


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

    # manually_write_aa_atoms_to_data_dir(path='../data/jsons/aa_atoms.json')  # This only needs to be done once.

    translate_aa_to_atoms(uniprot_ids=None)