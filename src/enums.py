from enum import Enum


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'            # ENUMERATED RESIDUES. (EQUIVALENT TO `ntcodes` IN DJ's ORIGINAL RNA CODE).
    ATOM_LABEL_NUM = 'atom_label_num'        # ENUMERATED ATOMS. (EQUIVALENT TO `atomcodes` IN DJ's ORIGINAL RNA CODE).
    AA_ATOM_PAIR = 'aa_atom_tuple'           # RESIDUE-ATOM PAIR (ONE TUPLE PER ROW).
    AA_ATOM_LABEL_NUM = 'aa_atom_label_num'  # ENUMERATED RESIDUE-ATOM PAIRS. (ALTERNATIVE WAY TO GENERATE `atomcodes`).
    BB_INDEX = 'bb_index'                    # POSITION OF 1 OUT OF 5 BACKBONE ATOMS. I'VE CHOSEN ALPHA-CARBON ('CA').
    BACKBONE_SIDECHAIN = 'bb_or_sc'          # BACKBONE OR SIDE-CHAIN ATOM ('bb' or 'sc').
    MEAN_COORDS = 'mean_xyz'                 # MEAN OF X, Y, Z, COORDINATES FOR EACH ATOM.
    MEAN_CORR_X = 'mean_corrected_x'         # X COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.
    MEAN_CORR_Y = 'mean_corrected_y'         # Y COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.
    MEAN_CORR_Z = 'mean_corrected_z'         # Z COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.


class CIF(Enum):
    # I INCLUDE THESE TWO PREFIXES TO KEEP TRACK OF THE PROVENANCE OF THE VARIOUS FIELDS.
    # IT'S NOT NECESSARY BUT I'M LEAVING IT THERE FOR NOW:
    S = '_pdbx_poly_seq_scheme.'
    A = '_atom_site.'

    S_seq_id = 'S_seq_id'                   # RESIDUE POSITION*
    S_mon_id = 'S_mon_id'                   # RESIDUE (3-LETTER)
    S_pdb_seq_num = 'S_pdb_seq_num'         # RESIDUE POSITION*
    S_asym_id = 'S_asym_id'                 # CHAIN

    A_group_PDB = 'A_group_PDB'             # GROUP, ('ATOM' or 'HETATM')
    A_id = 'A_id'                           # ATOM POSITION*
    A_label_atom_id = 'A_label_atom_id'     # ATOM
    A_label_comp_id = 'A_label_comp_id'     # RESIDUE (3-LETTER)
    A_label_asym_id = 'A_label_asym_id'     # CHAIN
    A_label_seq_id = 'A_label_seq_id'       # RESIDUE POSITION*
    A_Cartn_x = 'A_Cartn_x'                 # X COORDS
    A_Cartn_y = 'A_Cartn_y'                 # Y COORDS
    A_Cartn_z = 'A_Cartn_z'                 # Z COORDS
    A_occupancy = 'A_occupancy'             # OCCUPANCY

    HETATM = 'HETATM'
    ALPHA_CARBON = 'CA'                     # STRING USED TO REPRESENT ALPHA-CARBON OF POLYPEPTIDE CHAIN

# * ALL OF THESE POSITION INDICES ARE NEVER LESS THAN 1 (I.E. NEVER 0)


"""
  H    H    O*   H
  |    |    ||   | 
  N* - C* - C* - O*
  |    |
  H    R

Backbone is from the central horizontal atoms in figure above, as well as the double-bonded Oxygen shown in vertical.
Backbone atoms are highlighted with asterisks. 
The first Carbon, from the amino-terminal end is the alpha-Carbon 
which is therefore bound to the amino group H2N and to the side-chain group R.

Using format from EMBL-EBI website, the backbone is:

N - CA - C = 0 and OXT
 
'OXT' which is the C-terminal Oxygen that only occurs once in the polypeptide chain.



"""


# AMINO ACID ATOMS USING FORMAT FROM EMBL-EBI WEBSITE. (EXCLUDING INDIVIDUAL HYDROGENS).
class PolypeptideAtoms(Enum):

    BACKBONE = ('C', 'CA', 'N', 'O', 'OXT')  # AKA "MAIN-CHAIN"
    PEPTIDE_BOND_BACKBONE = ('C', 'N', 'O')  # SUBSET OF BACKBONE. PEPTIDE-BOND CARBONYL AND AMINO NITROGEN.
    SIDECHAIN = ('CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2',
                       'CZ3', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2', 'OG',
                       'OG1', 'OG2', 'OH', 'SD', 'SG')
