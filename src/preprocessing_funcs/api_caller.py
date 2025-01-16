"""
This module serves as a client for any API calls that are needed.
E.g. GET requests to 'https://files.rcsb.org/download/' and 'https://www.uniprot.org/uniprot/'.

"""
import requests


def call_for_cif_with_pdb_id(pdb_id: str) -> requests.Response:
    """
    Send GET request to https://files.rcsb.org/download/{pdb_id} with PDB identifier of interest.
    :param pdb_id: Alphanumeric 4-character Protein Databank Identifier. e.g. '1OJ6'.
    :return: Response code 200 and text data for given PDB id, or error code such as 404.
    """
    response = None
    pdb_id = pdb_id.upper()  # MUST BE UPPER-CASE
    pdb_id = pdb_id.removesuffix('.cif')
    url = f'https://files.rcsb.org/download/{pdb_id}.cif'

    try:
        print(f'Sending GET request to {url}')
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Failed to retrieve data from API: {e}')
    except Exception:
        print(f"Undefined error while trying to fetch '{pdb_id}' from PDB.")

    return response


def call_for_fasta_with_fasta_id(accession_id: str) -> requests.Response:
    """
    Send GET request to 'https://www.uniprot.org/uniprot/{fasta_id}' with UniProt identifier of interest.
    :param accession_id: UniProt 'Accession' identifier. Usually a 6- or 10-character alphanumeric code, e.g. 'P12345'.
    :return: Response code 200 and text data for given id, or error code such as 404.
    """
    response = None
    accession_id = accession_id.upper()  # SHOULD BE UPPER-CASE
    accession_id = accession_id.removesuffix('.fasta')
    url = f'https://www.uniprot.org/uniprot/{accession_id}.fasta'

    try:
        print(f'Sending GET request to {url}')
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Failed to retrieve data from API: {e}')
    except Exception:
        print(f"Undefined error while trying to call UniProt with accession fasta id = {accession_id}'.")

    return response
