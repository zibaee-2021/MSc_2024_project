# cuda_12.5.r12.5

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import pandas as pd


if __name__ == '__main__':
    pdb = '../data/4hb1.pdb'
    cif = '../data/4hb1.cif'

    dic = MMCIF2Dict(cif)
    df = pd.DataFrame.from_dict(dic, orient='index')
    df = df.transpose()

    with open(pdb, 'r') as f:
        fi = f.read()

    bla = pd.read_csv(pdb)
    sh = bla.shape
    print('h')


'group_PDB'
'id'
'label_atom_id'
'label_comp_id'
'label_asym_id'
'auth_seq_id'
'Cartn_x'
'Cartn_y'
'Cartn_z'
'occupancy'
'B_iso_or_equiv'
'type_symbol'