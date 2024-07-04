import json
import fasta_reader as reader
import faa_to_atoms

# INFERENCE: input is amino acid sequence --> output should be all possible atoms (obviously with no coords).

if __name__ == '__main__':

    id_to_atomic_sequence = faa_to_atoms.translate_aa_to_atoms(uniprot_ids=['P02185', 'PRS57'])
