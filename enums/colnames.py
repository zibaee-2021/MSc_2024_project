from enum import Enum


class ColNames(Enum):
    AA_LABEL_NUM = 'aa_label_num'  # `A_label_comp_id` enumerated (the amino acid)
    ATOM_LABEL_NUM = 'atom_label_num'  # `A_label_atom_id` enumerated (the atom)
    BB_INDEX = 'bb_index'  # NOT CLEAR WHAT THIS IS .. BACKBONE ATOMS ?  ?  ?
    MEAN_COORDS = 'mean_xyz'  # mean of x y z coordinates for each atom
    MEAN_CORR_X = 'mean_corrected_x'  # x coordinates for each atom subtracted by the mean of xyz coordinates
    MEAN_CORR_Y = 'mean_corrected_y'  # (as above) but for y coordinates
    MEAN_CORR_Z = 'mean_corrected_z'  # (as above) but for z coordinates
