from enum import Enum


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'       # ENUMERATED RESIDUES. (EQUIVALENT TO `ntcodes` IN DJ's ORIGINAL RNA CODE).
    ATOM_LABEL_NUM = 'atom_label_num'   # ENUMERATED ATOMS. (EQUIVALENT TO `atomcodes` IN DJ's ORIGINAL RNA CODE).
    BB_INDEX = 'bb_index'               # POSITION OF 1 OUT OF THE 5 BACKBONE ATOMS. I'VE CHOSEN ALPHA CARBON ('CA').
    MEAN_COORDS = 'mean_xyz'            # MEAN OF X, Y, Z, COORDINATES FOR EACH ATOM.
    MEAN_CORR_X = 'mean_corrected_x'    # X COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.
    MEAN_CORR_Y = 'mean_corrected_y'    # Y COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.
    MEAN_CORR_Z = 'mean_corrected_z'    # Z COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES.


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
    A_auth_seq_id = 'A_auth_seq_id'         # RESIDUE POSITION*
    A_Cartn_x = 'A_Cartn_x'                 # X COORDS
    A_Cartn_y = 'A_Cartn_y'                 # Y COORDS
    A_Cartn_z = 'A_Cartn_z'                 # Z COORDS
    A_occupancy = 'A_occupancy'             # OCCUPANCY

    HETATM = 'HETATM'

# * ALL OF THESE POSITION INDICES ARE NEVER LESS THAN 1 (I.E. NEVER 0)
