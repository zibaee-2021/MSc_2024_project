"""
Create enumerated collections of the amino acids, the atoms and the atoms for each amino acid.
Store in dicts and save to `../data/aa_atoms_enumerated`.

(Atom names manually collected from https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/list/ARG to TRP)
"""
from data_layer import data_handler as dh


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
    dh.write_to_jsons(aas_enumerated, atoms_only_enumerated)


if __name__ == '__main__':
    # This only needs to be done once:
    write_enumerated_atoms_and_residues()
