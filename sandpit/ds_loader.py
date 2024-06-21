# cuda_12.5.r12.5

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import pandas as pd


if __name__ == '__main__':
    pdb = '../data/4hb1.pdb'
    cif = '../data/4hb1.cif'

    mmcif_dict = MMCIF2Dict(cif)

    id_list = mmcif_dict['_atom_site.id']
    label_atom_id_list = mmcif_dict['_atom_site.label_atom_id']
    label_comp_id_list = mmcif_dict['_atom_site.label_comp_id']
    label_asym_id_list = mmcif_dict['_atom_site.label_asym_id']
    auth_seq_id_list = mmcif_dict['_atom_site.auth_seq_id']
    x_list = mmcif_dict['_atom_site.Cartn_x']
    y_list = mmcif_dict['_atom_site.Cartn_y']
    z_list = mmcif_dict['_atom_site.Cartn_z']
    occupancy_list = mmcif_dict['_atom_site.occupancy']
    B_iso_or_equiv_list = mmcif_dict['_atom_site.B_iso_or_equiv']
    type_symbol_list = mmcif_dict['_atom_site.type_symbol']

    d = {'id': id_list,
         'label_atom_id': label_atom_id_list,
         'label_comp_id': label_comp_id_list,
         'label_asym_id': label_asym_id_list,
         'auth_seq_id': auth_seq_id_list,
         'Cartn_x': x_list,
         'Cartn_y': y_list,
         'Cartn_z': z_list,
         'occupancy': occupancy_list,
         'B_iso_or_equiv': B_iso_or_equiv_list,
         'type_symbol': type_symbol_list}

    pdf = pd.DataFrame(data=d)
    pdf.head()
    sh = pdf.shape
    pass



