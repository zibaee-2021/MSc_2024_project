"""
Read local FASTA sequence files of proteins of interest, downloaded from UniProt.
"""
import os
import glob
from Bio import SeqIO
from io import StringIO
import api_caller as api


def read_fasta_sequences(uniprot_ids=None) -> dict:
    """
    Read specified FASTA sequences in.
    If FASTA file for the given UniProt id(s) is not found locally in `../data/FASTA/`, an attempt will be made to
    read it directly from Uniprot API webserver.
    If no identifier passed in, all FASTA files will be read from `../data/FASTA/`.
    :param uniprot_ids: A list of unique Uniprot identifiers (aka 'primary accession numbers').
    :return: 1-letter FASTA amino acid sequence(s) mapped to the identifier(s).
    e.g. {}

    """
    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]

    fasta_id_seqs = {}

    if not uniprot_ids:
        uniprot_ids = []
        print(f"No FASTA files were specified, "
              f"so all FASTA files found at '../../data/FASTA' will be read in.")
        fasta_files = glob.glob(os.path.join('../../data/FASTA', '*'))
        for fasta_file in fasta_files:
            fasta_id_faa = os.path.basename(fasta_file)
            fasta_id = os.path.splitext(fasta_id_faa)[0]
            uniprot_ids.append(fasta_id)

    for fasta_id in uniprot_ids:
        fasta_path = os.path.join('../../data/FASTA', fasta_id)
        fasta_path_faa = f'{fasta_path}.faa'

        try:
            record = SeqIO.read(fasta_path_faa, format='fasta')
            fasta_id_seqs[fasta_id] = str(record.seq)

        except FileNotFoundError:

            print(f"File '{fasta_path_faa}' not found")
            print(f'Will try to read {fasta_id} directly from Uniprot site..')
            response = api.call_for_fasta_with_fasta_id(fasta_id)
            fasta_data = StringIO(response.text)
            record = SeqIO.read(fasta_data, 'fasta')
            fasta_id_seqs[fasta_id] = str(record.seq)

        except Exception:
            print(f"Undefined error while trying to read in {fasta_path_faa}'.")

    return fasta_id_seqs


if __name__ == '__main__':
    Serine_Protease_Human_acc_id = 'Q6UWY2'
    fasta_id_seqs_ = read_fasta_sequences(uniprot_ids=Serine_Protease_Human_acc_id)
    print(fasta_id_seqs_)

