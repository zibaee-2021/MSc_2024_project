"""
NOTE: THE THREE FUNCTIONS IN THIS MODULE NEED ONLY BE CALLED **ONCE**.

Create enumerated collections of the amino acids, the atoms and the atoms for each amino acid.
Store in dicts and save to `../data/enumerations`.

(Atom names MANUALLY collected from https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/list/ARG to TRP)
"""
import json
import os
from enum import Enum
import ast
from typing import Tuple
from data_layer import data_handler as dh


class Path(Enum):
    enumeration_h_list = 'enumerations/hydrogens.lst'


class Filename(Enum):
    aa_atoms = 'residues_atoms'
    atoms = 'unique_atoms_only'
    aa = 'residues'
    atoms_no_h = 'unique_atoms_only_no_hydrogens'
    aa_atoms_no_h = 'residues_atoms_no_hydrogens'


def __enumerate_atoms_and_residues() -> Tuple[dict, dict, dict]:
    """
    ONLY CALLED ONCE TO GENERATE DICTS OF ENUMERATION MAPPINGS FOR ATOMS AND RESIDUE-ATOM PAIRS, TO BE USED FOR
    WRITING TO JSON FILES THAT ARE TO BE READ AND USED BY TOKENISER.PY.
    """
    residues, residues_atoms = dh.read_aa_atoms_yaml()
    residues_enumerated = {aa: i for i, aa in enumerate(residues)}

    aa_atom_idx = -1
    residues_atoms_enumerated = dict()
    unique_atoms_only = set()

    for aa, aa_atoms in residues_atoms.items():
        unique_atoms_only.update(aa_atoms)
        for aa_atom in aa_atoms:
            aa_atom_idx += 1
            residues_atoms_enumerated[(aa, aa_atom)] = aa_atom_idx
    residues_atoms_enumerated = {str(k): v for k, v in residues_atoms_enumerated.items()}

    unique_atoms_only = list(unique_atoms_only)
    unique_atoms_only.sort()
    nums = list(range(len(unique_atoms_only)))
    atoms_enumerated = {atom: num for atom, num in zip(unique_atoms_only, nums)}

    return residues_atoms_enumerated, atoms_enumerated, residues_enumerated


def _write_enumerated_atoms_and_residues() -> None:
    """
    ONLY CALLED ONCE TO GENERATE ENUMERATION MAPPINGS FOR ATOMS AND RESIDUE-ATOM PAIRS, THEN WRITTEN TO JSON FILES
    THAT ARE TO BE READ AND USED BY TOKENISER.PY.
    """
    residues_atoms_enumerated, atoms_enumerated, residues_enumerated = __enumerate_atoms_and_residues()
    dh.write_enumerations_json(fname=Filename.aa_atoms.value, dict_to_write=residues_atoms_enumerated)
    dh.write_enumerations_json(fname=Filename.atoms.value, dict_to_write=atoms_enumerated)
    dh.write_enumerations_json(fname=Filename.aa.value, dict_to_write=residues_enumerated)


def _write_enumerated_atoms_without_hydrogens() -> None:
    """
    ONLY CALLED ONCE TO GENERATE ENUMERATION MAPPINGS FOR ATOMS AND RESIDUE-ATOM PAIRS, WITHOUT HYDROGENS, THEN
    WRITTEN TO JSON FILES THAT ARE TO BE READ AND USED BY TOKENISER.PY.
    """
    hydrogen_atoms = dh.read_lst_file_from_data_dir(Path.enumeration_h_list.value)

    def __read_enumeration_mappings():
        _residues_atoms_enumerated = dh.read_enumerations_json(fname=Filename.aa_atoms.value)
        # _residues_atoms_enumerated = {eval(k): v for k, v in _residues_atoms_enumerated.items()}
        _atoms_enumerated = dh.read_enumerations_json(fname=Filename.atoms.value)
        _residues_enumerated = dh.read_enumerations_json(fname=Filename.aa.value)
        return _residues_atoms_enumerated, _atoms_enumerated, _residues_enumerated

    residues_atoms_enumerated, unique_atoms_only_enumerated, _residues_unused = __read_enumeration_mappings()

    unique_atoms_except_h = [k for k in unique_atoms_only_enumerated if k not in hydrogen_atoms]
    for atom in unique_atoms_except_h:
        assert not atom.startswith('H'), (f"Atom '{atom}' starts with 'H', which is a Hydrogen, so the code or "
                                          f"hydrogens.lst file must be amended so that it gets removed above.")
    assert len(hydrogen_atoms) + len(unique_atoms_except_h) == len(unique_atoms_only_enumerated)
    nums = list(range(len(unique_atoms_except_h)))
    unique_atoms_only_no_hydrogens = {atom: num for atom, num in zip(unique_atoms_except_h, nums)}
    dh.write_enumerations_json(fname=Filename.atoms_no_h.value, dict_to_write=unique_atoms_only_no_hydrogens)

    residues_atoms_except_h = dict()
    num = 0
    for key in residues_atoms_enumerated:
        k = ast.literal_eval(key)
        atm = k[1]
        if atm not in hydrogen_atoms:
            residues_atoms_except_h[key] = num
            num += 1

    for key in residues_atoms_except_h:
        k = ast.literal_eval(key)
        atm = k[1]
        assert not atm.startswith('H'), (f"Atom '{atm}' starts with 'H', which is a Hydrogen, so the code or "
                                         f"hydrogens.list file must be amended so that it gets removed above.")
    dh.write_enumerations_json(fname=Filename.aa_atoms_no_h.value, dict_to_write=residues_atoms_except_h)


# if __name__ == '__main__':
    # # THESE NEED ONLY BE RUN ONCE, TO GENERATE THE JSON FILE(S):
    # # This only needs to be done once:
    # _write_enumerated_atoms_and_residues()
    # # This only needs to be done once:
    # _write_enumerated_atoms_without_hydrogens()
