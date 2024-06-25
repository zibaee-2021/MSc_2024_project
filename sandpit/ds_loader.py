# cuda_12.5.r12.5

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import pandas as pd
from enum import Enum

train_list = []
validation_list = []
tnum = 0


if __name__ == '__main__':

    class mmCIFAttributes(Enum):
        S_seq_id = 'S_seq_id'


    pdb = '../data/4hb1.pdb'
    cif = '../data/4hb1.cif'

    mmcif_dict = MMCIF2Dict(cif)

    asym_id_list = mmcif_dict['_pdbx_poly_seq_scheme.asym_id']
    entity_id_list = mmcif_dict['_pdbx_poly_seq_scheme.entity_id']
    seq_id_list = mmcif_dict['_pdbx_poly_seq_scheme.seq_id']
    mon_id_list = mmcif_dict['_pdbx_poly_seq_scheme.mon_id']
    ndb_seq_num_list = mmcif_dict['_pdbx_poly_seq_scheme.ndb_seq_num']
    pdb_seq_num_list = mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num']
    auth_seq_num_list = mmcif_dict['_pdbx_poly_seq_scheme.auth_seq_num']
    pdb_mon_id_list = mmcif_dict['_pdbx_poly_seq_scheme.pdb_mon_id']
    auth_mon_id_list = mmcif_dict['_pdbx_poly_seq_scheme.auth_mon_id']
    pdb_strand_id_list = mmcif_dict['_pdbx_poly_seq_scheme.pdb_strand_id']
    pdb_ins_code_list = mmcif_dict['_pdbx_poly_seq_scheme.pdb_ins_code']
    hetero_list = mmcif_dict['_pdbx_poly_seq_scheme.hetero']

    # I'm using 'S_' prefix for the `_pdbx_poly_seq_scheme` source
    # d1 = {'S_asym_id': asym_id_list,
    #       'S_entity_id': entity_id_list,
    d1 = {'S_seq_id': seq_id_list,
          'S_mon_id': mon_id_list,
          'S_ndb_seq_num': ndb_seq_num_list,
          'S_pdb_seq_num': pdb_seq_num_list,
          'S_auth_seq_num': auth_seq_num_list,
          'S_pdb_mon_id': pdb_mon_id_list,
          'S_auth_mon_id': auth_mon_id_list,
          'S_pdb_strand_id': pdb_strand_id_list,
          # 'S_pdb_ins_code': pdb_ins_code_list,
          # 'S_hetero': hetero_list
          }

    pdf_left = pd.DataFrame(data=d1)

    id_list = mmcif_dict['_atom_site.id']
    label_atom_id_list = mmcif_dict['_atom_site.label_atom_id']  #
    label_comp_id_list = mmcif_dict['_atom_site.label_comp_id']  # aa 3-letter
    label_asym_id_list = mmcif_dict['_atom_site.label_asym_id']  # chain
    auth_seq_id_list = mmcif_dict['_atom_site.auth_seq_id']
    x_list = mmcif_dict['_atom_site.Cartn_x']
    y_list = mmcif_dict['_atom_site.Cartn_y']
    z_list = mmcif_dict['_atom_site.Cartn_z']
    occupancy_list = mmcif_dict['_atom_site.occupancy']
    B_iso_or_equiv_list = mmcif_dict['_atom_site.B_iso_or_equiv']
    type_symbol_list = mmcif_dict['_atom_site.type_symbol']

    # I'm using 'A_' prefix for the `_atom_site` source
    d2 = {'A_id': id_list,
          'A_label_atom_id': label_atom_id_list,
          'A_label_comp_id': label_comp_id_list,
          # 'label_asym_id': label_asym_id_list,  # chain
          'A_auth_seq_id': auth_seq_id_list,
          'A_Cartn_x': x_list,
          'A_Cartn_y': y_list,
          'A_Cartn_z': z_list,
          'A_occupancy': occupancy_list,
          # 'A_B_iso_or_equiv': B_iso_or_equiv_list,
          'A_type_symbol': type_symbol_list,
          }

    pdf_right = pd.DataFrame(data=d2)

    pdf_merged = pd.merge(left=pdf_left, right=pdf_right, left_on='S_auth_seq_num', right_on='A_auth_seq_id',
                          how='outer')

    for col in ['A_Cartn_x', 'A_Cartn_y', 'A_Cartn_z', 'A_occupancy']:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')

    for col in ['S_seq_id', 'S_ndb_seq_num', 'S_pdb_seq_num', 'S_auth_seq_num', 'A_id', 'A_auth_seq_id']:
        pdf_merged[col] = pd.to_numeric(pdf_merged[col], errors='coerce')
        pdf_merged[col] = pdf_merged[col].astype('Int64')

    cols = pdf_merged.columns
    print(cols)
    col_dtypes = pdf_merged.dtypes
    print(col_dtypes)

    # pdf2.loc[pdf['occupancy'] >= 0.5]

    pdf_merged = pdf_merged['S_auth_seq_num', 'A_auth_seq_id', 'A_id', 'A_label_atom_id', 'A_type_symbol',
                            'S_pdb_seq_num', 'S_seq_id', 'S_ndb_seq_num', 'S_mon_id', 'S_pdb_mon_id', 'S_auth_mon_id',
                            'A_label_comp_id', 'S_pdb_strand_id', 'A_Cartn_x ', 'A_Cartn_y',  'A_Cartn_z',
                            'A_occupancy']









    # # converting dj variable names
    # d = {'atid': id_list,
    #      'atomcodes': label_atom_id_list,
    #      'aacode': label_comp_id_list,
    #      'chain': label_asym_id_list,
    #      'auth_seq_id': auth_seq_id_list,
    #      'Cartn_x': x_list,
    #      'Cartn_y': y_list,
    #      'Cartn_z': z_list,
    #      'occupancy': occupancy_list,
    #      'B_iso_or_equiv': B_iso_or_equiv_list,
    #      'type_symbol': type_symbol_list,
    #      '_entity_poly_seq.num': _entity_poly_seq.num,
    #      '_entity_poly_seq.mon_id': _entity_poly_seq.mon_id
    #      }
    #



