import os
from unittest import TestCase
import pandas as pd
from data_layer import data_handler as dh
from src.preprocessing_funcs import data_validator as dv


class TestDataHandler(TestCase):

    def test_read_json_from_data_dir(self):
        fname = 'enumeration/residues.json'
        f = dh._read_json_from_data_dir(fname)
        expected = {'ALA': 0,
                    'CYS': 1,
                    'ASP': 2,
                    'GLU': 3,
                    'PHE': 4,
                    'GLY': 5,
                    'HIS': 6,
                    'ILE': 7,
                    'LYS': 8,
                    'LEU': 9,
                    'MET': 10,
                    'ASN': 11,
                    'PRO': 12,
                    'GLN': 13,
                    'ARG': 14,
                    'SER': 15,
                    'THR': 16,
                    'VAL': 17,
                    'TRP': 18,
                    'TYR': 19}
        self.assertDictEqual(expected, f)

    def test_read_lst_file_from_data_dir(self):
        fname = 'enumeration/hydrogens.lst'
        f = dh.read_lst_file_from_data_dir(fname)
        expected = ['H', 'H2', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD3', 'HD11', 'HD12',
                    'HD13', 'HD21', 'HD22', 'HD23', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG2',
                    'HG3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HH', 'HH2', 'HH11', 'HH12', 'HH21', 'HH22',
                    'HXT', 'HZ', 'HZ1', 'HZ2', 'HZ3']
        self.assertListEqual(expected, f)

    def test_check_protein_and_atom_numbering_of_parsed_cif(self):
        print(os.getcwd())
        tokenised_cif = pd.read_csv('test_data/1OJ6_test.ssv', sep=' ')
        tokenised_cifs = dict()
        tokenised_cifs['1OJ6_test'] = tokenised_cif
        dv.check_protein_and_atom_numbering_of_parsed_tokenised_cif_ssv(tokenised_cifs)
        expected = ''
        actual = ''
        self.assertEqual(expected, actual)
