#!/usr/bin/env python3
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.MMCIFParser import FastMMCIFParser

argv = sys.argv
argc = len(argv)

if argc < 3:
    print('Usage: cif2pdb input.cif output.pdb')
    sys.exit(0)

cif = argv[1]
pdb = argv[2]
    
parser = FastMMCIFParser()  # Only read ATOM and HETATM records
pdbio = PDBIO()


structure = parser.get_structure(structure_id='4hb1', filename=cif)
pdbio.set_structure(structure)

# the save method has a parameter `select` which can be used to filter atoms/residues/chains etc.
pdbio.save(file=pdb, write_end=True, preserve_atom_numbering=False)


