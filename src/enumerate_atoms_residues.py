"""
Create enumerated collections of the amino acids, the atoms and the atoms for each amino acid.
Store in dicts and save to `../data/jsons`.

(Atom names manually collected from https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/list/ARG to TRP)
"""
import yaml
from typing import Tuple
import json


def _read_atom() -> Tuple[list, dict]:
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


def _write_to_jsons(aas_enumerated, atoms_only_enumerated):
    with open('../data/jsons/aas_enumerated.json', 'w') as json_f:
        json.dump(aas_enumerated, json_f, indent=4)

    # aas_atoms_enumerated_ = {str(k): v for k, v in aas_atoms_enumerated.items()}
    # with open('../data/jsons/aas_atoms_enumerated.json', 'w') as json_f:
    #     json.dump(aas_atoms_enumerated_, json_f, indent=4)

    with open('../data/jsons/unique_atoms_only_enumerated.json', 'w') as json_f:
        json.dump(atoms_only_enumerated, json_f, indent=4)


def _enumerate_atoms_and_residues():
    residues, residues_atoms = _read_atom()
    aas_enumerated = {aa: i for i, aa in enumerate(residues)}

    aa_atom_idx = -1
    aas_atoms_enumerated = dict()
    unique_atoms_only = set()

    for aa, aa_atoms in residues_atoms.items():
        unique_atoms_only.update(aa_atoms)
        # for aa_atom in aa_atoms:
        #     aa_atom_idx += 1
        #     aas_atoms_enumerated[(aa, aa_atom)] = aa_atom_idx

    unique_atoms_only = list(unique_atoms_only)
    unique_atoms_only.sort()
    nums = list(range(len(unique_atoms_only)))
    atoms_only_enumerated = {atom: num for atom, num in zip(unique_atoms_only, nums)}
    return aas_enumerated, atoms_only_enumerated


def write_enumerated_atoms_and_residues():
    aas_enumerated, atoms_only_enumerated = _enumerate_atoms_and_residues()
    _write_to_jsons(aas_enumerated, atoms_only_enumerated)


if __name__ == '__main__':
    # This only needs to be done once:
    write_enumerated_atoms_and_residues()
