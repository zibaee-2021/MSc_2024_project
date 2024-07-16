# Placeholder file for script that will contain all the functions that read/write from/to `data` subdirs, to just make
# the other functions a bit tidier.
import glob
import os
import json

import torch
import yaml
from typing import Tuple
from src import api_caller as api


def _chdir_to_dh():
    """
    Store current working directory, in order to revert back to it after finishing
    :return: Current working directory *before* changing it to the local dir of data_handler.
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f'Path changed from {cwd} temporarily to = {os.getcwd()}')
    return cwd


def read_list_of_pdbids_from_text_file(filename: str):
    cwd = _chdir_to_dh()
    path = '../data/pdb_ids_list'
    path_file = os.path.join(path, filename)
    with open(path_file, 'r') as f:
        pdb_ids = f.read()
    pdbids = pdb_ids.split()
    os.chdir(cwd)
    return pdbids


def get_list_of_pdbids_of_local_single_domain_cifs() -> list:
    cwd = _chdir_to_dh()
    cifs = glob.glob(os.path.join('../data/cifs_single_domain_prots', '*.cif'))
    path_cifs = [cif for cif in cifs if os.path.isfile(cif)]
    pdb_ids = []

    for path_cif in path_cifs:
        cif_basename = os.path.basename(path_cif)
        pdbid = os.path.splitext(cif_basename)[0]
        pdb_ids.append(pdbid)

    os.chdir(cwd)
    return pdb_ids


# def get_list_of_uniprotids_of_locally_downloaded_cifs():


def write_list_to_space_separated_txt_file(list_to_write: list, file_name: str) -> None:
    cwd = _chdir_to_dh()
    space_sep_str = ' '.join(list_to_write)
    with open(f'../data/{file_name}', 'w') as f:
        f.write(space_sep_str)
    os.chdir(cwd)


def manually_write_aa_atoms_to_data_dir(path: str) -> None:
    """
    This function only needs to be run once.
    :param path:
    :return:
    """
    cwd = _chdir_to_dh()
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
    os.chdir(cwd)


def read_fasta_aa_enumeration_mapping() -> dict:
    cwd = _chdir_to_dh()
    with open('../data/aa_atoms_enumerated/FASTA_aas_enumerated.json', 'r') as json_f:
        fasta_aas_enumerated = json.load(json_f)
    os.chdir(cwd)
    return fasta_aas_enumerated


def read_3letter_aas_enumerated_mapping() -> dict:
    cwd = _chdir_to_dh()
    with open('../data/aa_atoms_enumerated/aas_enumerated.json', 'r') as json_f:
        aas_enumerated = json.load(json_f)
    os.chdir(cwd)
    return aas_enumerated


def read_atom_enumeration_mapping() -> dict:
    cwd = _chdir_to_dh()
    with open('../data/aa_atoms_enumerated/unique_atoms_only_enumerated.json', 'r') as json_f:
        atoms_enumerated = json.load(json_f)
    os.chdir(cwd)
    return atoms_enumerated


def read_enumeration_mappings() -> Tuple[dict, dict, dict]:
    cwd = _chdir_to_dh()
    atoms_enumerated = read_atom_enumeration_mapping()
    aas_enumerated = read_3letter_aas_enumerated_mapping()
    fasta_aas_enumerated = read_fasta_aa_enumeration_mapping()
    os.chdir(cwd)
    return atoms_enumerated, aas_enumerated, fasta_aas_enumerated


def write_to_jsons(aas_enumerated, atoms_only_enumerated):
    cwd = _chdir_to_dh()
    with open('../data/aa_atoms_enumerated/aas_enumerated.json', 'w') as json_f:
        json.dump(aas_enumerated, json_f, indent=4)

    # aas_atoms_enumerated_ = {str(k): v for k, v in aas_atoms_enumerated.items()}
    # with open('../data/aa_atoms_enumerated/aas_atoms_enumerated.json', 'w') as json_f:
    #     json.dump(aas_atoms_enumerated_, json_f, indent=4)

    with open('../data/aa_atoms_enumerated/unique_atoms_only_enumerated.json', 'w') as json_f:
        json.dump(atoms_only_enumerated, json_f, indent=4)
    os.chdir(cwd)


def read_aa_atoms_yaml() -> Tuple[list, dict]:
    cwd = _chdir_to_dh()
    aas_atoms = dict()
    aas = list()

    with open('../data/yamls/atoms_residues.yaml', 'r') as stream:
        try:
            atoms_aas = yaml.load(stream, Loader=yaml.Loader)
            aas = atoms_aas['ROOT']['AAs']
            aas_atoms = atoms_aas['ROOT']['ATOMS_BY_AA']

        except yaml.YAMLError as exc:
            print(exc)
    os.chdir(cwd)
    return aas, aas_atoms


def write_pdb_uniprot_fasta_recs_to_json(recs: dict, filename: str) -> None:
    cwd = _chdir_to_dh()
    with open(f'../data/FASTA/{filename}.json', 'w') as json_f:
        json.dump(recs, json_f, indent=4)
    os.chdir(cwd)


def _remove_null_entries(pdbids_fasta_json: dict):
    pdbids_fasta_json = {k: v for k, v in pdbids_fasta_json.items() if v is not None}
    return pdbids_fasta_json


def read_nonnull_fastas_from_json_to_dict(filename: str) -> dict:
    cwd = _chdir_to_dh()
    with open(f'../data/FASTA/{filename}.json', 'r') as json_f:
        pdbids_fasta_json = json.load(json_f)
    pdbids_fasta_json = _remove_null_entries(pdbids_fasta_json)
    os.chdir(cwd)
    return pdbids_fasta_json


def fetch_mmcif_from_pdb_api_and_write_locally(pdb_ids: list, dst_path: str):
    cwd = _chdir_to_dh()
    non_200_count = 0
    for pdb_id in pdb_ids:

        mmcif_file = f'{dst_path}{pdb_id}.cif'
        if not os.path.exists(mmcif_file):
            response = api.call_for_cif_with_pdb_id(pdb_id)
            code = response.status_code
            if code != 200:
                non_200_count += 1
                print(f'Response status code for {pdb_id} is {code}, hence could not read the pdb for this id.')

            with open(mmcif_file, 'w') as file:
                file.write(response.text)

    print(f'{non_200_count} non-200 status codes out of {len(pdb_ids)} PDB API calls.')
    os.chdir(cwd)


def save_torch_tensor(pt: torch.Tensor, dst_path: str):
    cwd = _chdir_to_dh()
    torch.save(pt, f'{dst_path}.pt')
    os.chdir(cwd)


# if __name__ == '__main__':
# # This only needs to be run once:
#     dh.manually_write_aa_atoms_to_data_dir(path='../data/aa_atoms_enumerated/aa_atoms.json')

