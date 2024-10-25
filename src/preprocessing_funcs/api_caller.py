"""
This module serves as a client for any API calls that are needed.

"""
from enum import Enum
import requests


class Urls(Enum):
    PDB = 'https://files.rcsb.org/download/'
    UNIPROT = 'https://www.uniprot.org/uniprot/'


def call_for_cif_with_pdb_id(pdb_id: str) -> requests.Response:
    response = None
    url = f'{Urls.PDB.value}{pdb_id}.cif'

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Failed to retrieve data from API: {e}')
    except Exception:
        print(f"Undefined error while trying to fetch '{pdb_id}' from PDB.")

    return response


def call_for_fasta_with_fasta_id(fasta_id: str) -> requests.Response:
    response = None
    url = f'{Urls.UNIPROT.value}{fasta_id}.fasta'

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Failed to retrieve data from API: {e}')
    except Exception:
        print(f"Undefined error while trying to call Uniprot with fasta id = {fasta_id}'.")

    return response
