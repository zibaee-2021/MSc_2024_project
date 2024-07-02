from Bio.PDB import PDBParser

# Define the atomic composition of each amino acid
amino_acid_atoms = {
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


# Function to translate an amino acid sequence into its corresponding atomic sequence
def translate_to_atoms(sequence):
    atomic_sequence = []
    for residue in sequence:
        if residue in amino_acid_atoms:
            atomic_sequence.extend(amino_acid_atoms[residue])
        else:
            raise ValueError(f"Unknown amino acid: {residue}")
    return atomic_sequence


if __name__ == '__main__':

    # Example amino acid sequence
    amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWY'

    # Translate to atomic sequence
    atomic_sequence = translate_to_atoms(amino_acid_sequence)

    # Print the atomic sequence
    print(atomic_sequence)
