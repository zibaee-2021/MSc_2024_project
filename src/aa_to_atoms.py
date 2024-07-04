from Bio.PDB import PDBParser
import json
import fasta_reader as reader


def translate_to_atoms(aa_fasta_sequence: str) -> str:
    """
    Translate the given amino acid sequence to its corresponding atomic sequence using the PDB atom naming convention.
    :param aa_fasta_sequence: Amino acid sequence using 1-letter format.
    :return: The atomic sequence corresponding to the given amino acid sequence.
    """
    atomic_sequence = []
    for aa in aa_fasta_sequence:
        if aa in amino_acid_atoms:
            atomic_sequence.extend(amino_acid_atoms[aa])
        else:
            raise ValueError(f'{aa} is not one of the 20 amino acids (using 1-letter format)')
    return ''.join(atomic_sequence)


if __name__ == '__main__':

    # aa_atoms = {
    #     'A': ['N', 'CA', 'C', 'O', 'CB'],
    #     'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    #     'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    #     'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    #     'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    #     'G': ['N', 'CA', 'C', 'O'],
    #     'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    #     'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    #     'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    #     'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    #     'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    #     'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    #     'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    #     'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    #     'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    #     'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    #     'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    #     'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    #     'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    #     'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
    # }
    # with open('../data/jsons/aa_atoms.json', 'w') as json_f:
    #     json.dump(aa_atoms, json_f, indent=4)

    aa_atom_path = '../data/jsons/aa_atoms.json'
    try:
        with open(aa_atom_path, 'r') as json_f:
            amino_acid_atoms = json.load(json_f)
    except FileNotFoundError:
        print(f'{aa_atom_path} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")

    fasta_sequences = reader.read_fasta_sequences()

    atomic_sequences = []
    for fasta_sequence in fasta_sequences:
        atomic_seq = translate_to_atoms(aa_fasta_sequence=fasta_sequence)
        atomic_sequences.append(atomic_seq)
    print(atomic_sequences)
