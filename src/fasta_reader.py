"""
Read local fasta sequence files of proteins of interest, downloaded from UniProt.
"""

from Bio import SeqIO
from enum import Enum


class Sequences(Enum):
    Myoglobin_Sperm_whale = 'P02185'
    Serine_Protease_Human = 'PRS57'
    Low_resolution_structure = ''  # resolution > 4 angstroms
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)


def read_fasta_sequences():
    myoglob_fasta = SeqIO.read(f'../data/fastas/{Sequences.Myoglobin_Sperm_whale.value}.faa', format='fasta')
    print(f'Sperm whale Myoglobin sequence: {str(myoglob_fasta.seq)}'
          f'\nLength = {len(str(myoglob_fasta.seq))} amino acids')

    serine_protease_fasta = SeqIO.read(f'../data/fastas/{Sequences.Serine_Protease_Human.value}.faa', format='fasta')
    print(f'Human Serine Protease sequence: {str(serine_protease_fasta.seq)}'
          f'\nLength = {len(str(serine_protease_fasta.seq))} amino acids')

    return myoglob_fasta, serine_protease_fasta


if __name__ == '__main__':
    myoglob, ser_prot = read_fasta_sequences()