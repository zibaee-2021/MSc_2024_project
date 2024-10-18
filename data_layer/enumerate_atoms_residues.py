"""
Create enumerated collections of the amino acids, the atoms and the atoms for each amino acid.
Store in dicts and save to `../data/aa_atoms_enumerated`.

(Atom names manually collected from https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/list/ARG to TRP)
"""
from data_layer import data_handler as dh
import json
import os
import ast


def _enumerate_atoms_and_residues():
    residues, residues_atoms = dh.read_aa_atoms_yaml()
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
    dh.write_to_json_to_data_dir(fname='aa_atoms_enumerated/aas_enumerated.json', dict_to_write=aas_enumerated)
    dh.write_to_json_to_data_dir(fname='aa_atoms_enumerated/unique_atoms_only_enumerated.json', dict_to_write=atoms_only_enumerated)


def write_enumerated_atoms_without_hydrogens():
    hydrogen_atoms = dh.read_lst_file_from_data_dir('aa_atoms_enumerated/hydrogens.lst')

    unique_atoms_only_enumerated = dh.read_json_from_data_dir('/aa_atoms_enumerated/unique_atoms_only_enumerated.json')
    unique_atoms_except_h = [k for k in unique_atoms_only_enumerated if k not in hydrogen_atoms]
    for atom in unique_atoms_except_h:
        assert not atom.startswith('H'), (f"Atom '{atom}' starts with 'H', which is a Hydrogen, so the code or "
                                          f"hydrogens.list file must be amended so that it gets removed above.")
    assert len(hydrogen_atoms) + len(unique_atoms_except_h) == len(unique_atoms_only_enumerated)
    nums = list(range(len(unique_atoms_except_h)))
    unique_atoms_only_enumerated_no_hydrogens = {atom: num for atom, num in zip(unique_atoms_except_h, nums)}
    dh.write_to_json_to_data_dir(fname='aa_atoms_enumerated/unique_atoms_only_enumerated_no_hydrogens.json',
                                 dict_to_write=unique_atoms_only_enumerated_no_hydrogens)

    aas_atoms_enumerated = dh.read_json_from_data_dir('/aa_atoms_enumerated/aas_atoms_enumerated.json')

    aas_atoms_except_h = dict()
    num = 0
    for key in aas_atoms_enumerated:
        k = ast.literal_eval(key)
        atm = k[1]
        if atm not in hydrogen_atoms:
            aas_atoms_except_h[key] = num
            num += 1

    for key in aas_atoms_except_h:
        k = ast.literal_eval(key)
        atm = k[1]
        assert not atm.startswith('H'), (f"Atom '{atm}' starts with 'H', which is a Hydrogen, so the code or "
                                          f"hydrogens.list file must be amended so that it gets removed above.")
    dh.write_to_json_to_data_dir(fname='aa_atoms_enumerated/aas_atoms_enumerated_no_hydrogens.json',
                                 dict_to_write=aas_atoms_except_h)


if __name__ == '__main__':
    pass
    # # This only needs to be done once:
    # write_enumerated_atoms_and_residues()

    # # This only needs to be done once:
    # write_enumerated_atoms_without_hydrogens()