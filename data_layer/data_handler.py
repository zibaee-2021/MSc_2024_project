# Placeholder file for script that will contain all the functions that read/write from/to `data` subdirs, to just make
# the other functions a bit tidier.
import glob
import os
import json
import yaml
from typing import Tuple


os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(f'Path from data_handler= {os.getcwd()}')


def read_list_of_pdbids_from_text_file(filename: str):
    path = '../data/pdb_ids_list'
    path_file = os.path.join(path, filename)
    with open(path_file, 'r') as f:
        pdb_ids = f.read()
    pdbids = pdb_ids.split()
    return pdbids


def get_list_of_pdbids_of_local_single_domain_cifs() -> list:
    cifs = glob.glob(os.path.join('../data/cifs_single_domain_prots', '*.cif'))
    path_cifs = [cif for cif in cifs if os.path.isfile(cif)]

    pdb_ids = []

    for path_cif in path_cifs:
        cif_basename = os.path.basename(path_cif)
        pdbid = os.path.splitext(cif_basename)[0]
        pdb_ids.append(pdbid)

    return pdb_ids


# def get_list_of_uniprotids_of_locally_downloaded_cifs():


def write_list_to_space_separated_txt_file(list_to_write: list, file_name: str) -> None:
    space_sep_str = ' '.join(list_to_write)
    with open(f'../data/{file_name}', 'w') as f:
        f.write(space_sep_str)


def manually_write_aa_atoms_to_data_dir(path: str) -> None:
    """
    This function only needs to be run once.
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


def read_enumeration_mappings() -> Tuple[dict, dict]:
    with open('../data/aa_atoms_enumerated/unique_atoms_only_enumerated.json', 'r') as json_f:
        atoms_enumerated = json.load(json_f)
    with open('../data/aa_atoms_enumerated/aas_enumerated.json', 'r') as json_f:
        aas_enumerated = json.load(json_f)
    return atoms_enumerated, aas_enumerated


def write_to_jsons(aas_enumerated, atoms_only_enumerated):
    with open('../data/aa_atoms_enumerated/aas_enumerated.json', 'w') as json_f:
        json.dump(aas_enumerated, json_f, indent=4)

    # aas_atoms_enumerated_ = {str(k): v for k, v in aas_atoms_enumerated.items()}
    # with open('../data/aa_atoms_enumerated/aas_atoms_enumerated.json', 'w') as json_f:
    #     json.dump(aas_atoms_enumerated_, json_f, indent=4)

    with open('../data/aa_atoms_enumerated/unique_atoms_only_enumerated.json', 'w') as json_f:
        json.dump(atoms_only_enumerated, json_f, indent=4)


def read_aa_atoms_yaml() -> Tuple[list, dict]:
    aas_atoms = dict()
    aas = list()

    with open('../data/yamls/atoms_residues.yaml', 'r') as stream:
        try:
            atoms_aas = yaml.load(stream, Loader=yaml.Loader)
            aas = atoms_aas['ROOT']['AAs']
            aas_atoms = atoms_aas['ROOT']['ATOMS_BY_AA']

        except yaml.YAMLError as exc:
            print(exc)

    return aas, aas_atoms


def write_pdb_uniprot_fasta_recs_to_json(recs: dict, filename: str) -> None:
    with open(f'../data/FASTA/{filename}.json', 'w') as json_f:
        json.dump(recs, json_f, indent=4)


def read_fastas_from_json_to_dict(filename: str) -> dict:
    with open(f'../data/FASTA/{filename}.json', 'r') as json_f:
        pdbids_fasta_json = json.load(json_f)
    return pdbids_fasta_json


# if __name__ == '__main__':
# # This only needs to be run once:
#     dh.manually_write_aa_atoms_to_data_dir(path='../data/aa_atoms_enumerated/aa_atoms.json')

