import json
from enum import Enum
import fasta_reader as reader
import faa_to_atoms

# INFERENCE: input is amino acid sequence --> output should be all possible atoms (obviously with no coords).


# Ideally these should be UNIQUE identifiers for the corresponding proteins
class UniprotId(Enum):
    Myoglobin_pig = 'P02189'  # PDB '1mwd' pig WT deoxy myoglobin
    # Myoglobin_sperm_whale = 'P02185'  # PDB '104M'
    Serine_Protease_Human = 'Q6UWY2'  # Serine protease 57
    Low_resolution_structure = ''  # resolution > 4 angstroms ??
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)


if __name__ == '__main__':

    id_to_atomic_sequence = faa_to_atoms.translate_aa_to_atoms(uniprot_ids=[UniprotId.Myoglobin_pig.value,
                                                                            UniprotId.Serine_Protease_Human.value])
