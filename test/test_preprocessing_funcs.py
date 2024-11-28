import os, sys
import pandas as pd
from unittest import TestCase
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from src.preprocessing_funcs import tokeniser as tk
from src.preprocessing_funcs import cif_parser
from data_layer import data_handler as dh
from src.enums import CIF

"""
Expected 17 columns of output dataframe of `parse_tokenise_cif_write_flatfile()`:

A_label_asym_id       # CHAIN                 - JOIN ON THIS, SORT ON THIS, KEEP IN DF.
S_seq_id              # RESIDUE POSITION      - SORT ON THIS, KEEP IN DATAFRAME.
A_id                  # ATOM POSITION         - SORT ON THIS, KEEP IN DF.
A_label_atom_id       # ATOM                  - KEEP IN DF.
A_Cartn_x             # COORDINATES           - ATOM X-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_y             # COORDINATES           - ATOM Y-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
A_Cartn_z             # COORDINATES           - ATOM Z-COORDINATES (SUBSEQUENTLY CORRECTED BY MEAN), CAN BE REMOVED.
aa_label_num          # ENUMERATED RESIDUES   - EQUIVALENT TO `ntcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
bb_or_sc              # BACKBONE OR SIDE-CHAIN ATOM ('bb' or 'sc'), KEEP FOR POSSIBLE SUBSEQUENT OPERATIONS.
bb_atom_pos           # ATOM POSITION CA OR MOST C-TERM OTHER BB ATOM, PER RESIDUE. KEEP IN DF.
bbindices             # INDEX POSITION OF THE ATOM POSITION (`A_id`) OF ALLOCATED BACKBONE ATOM.
atom_label_num        # ENUMERATED ATOMS      - EQUIVALENT TO `atomcodes` IN ORIGINAL RNA CODE. KEEP IN DF.
aa_atom_tuple         # RESIDUE-ATOM PAIR     - ONE TUPLE PER ROW. KEEP IN DF.
aa_atom_label_num     # ENUMERATED RESIDUE-ATOM PAIRS. (ALTERNATIVE WAY TO GENERATE `atomcodes`).
mean_xyz              # MEAN OF COORDS        - MEAN OF X, Y, Z COORDINATES FOR EACH ATOM. KEEP IN DF TEMPORARILY.
mean_corrected_x      # X-COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_y      # Y-COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
mean_corrected_z      # Z-COORDINATES FOR EACH ATOM SUBTRACTED BY THE MEAN OF XYZ COORDINATES, ROW-WISE. KEEP IN DF.
"""


class TestPreprocessingFuncs(TestCase):

    def setUp(self):
        self.test_cif_dir = 'test_data/cif'
        self.test_tokenised_dir = 'test_data/tokenised'
        self.test_joined_dir = 'test_data/merged'
        self.test_1V5H_ssv = 'test_1V5H.ssv'
        self.test_1V5H_ssv_small = 'test_1V5H_small.ssv'
        self.test_1V5H_cif = 'test_1V5H.cif'
        self.test_1OJ6_4chains_cif = 'test_1OJ6_4chains.cif'

        not_yet_called = False
        # CALL ONLY ONCE (i.e. SET FLAG TO FALSE AFTER CALL)
        if not_yet_called:
            mmcif_dict = MMCIF2Dict(f'{self.test_cif_dir}/{self.test_1V5H_cif}')
            polyseq_pdf = cif_parser._extract_fields_from_poly_seq(mmcif_dict)
            atomsite_pdf = cif_parser._extract_fields_from_atom_site(mmcif_dict)
            atomsite_pdf = cif_parser._remove_hetatm_rows(atomsite_pdf)
            pdf_merged = cif_parser._join_atomsite_to_polyseq(atomsite_pdf, polyseq_pdf)
            os.makedirs(self.test_joined_dir, exist_ok=True)
            pdf_merged.to_csv(path_or_buf=f'{self.test_joined_dir}/{self.test_1V5H_ssv}', sep=' ', index=False)

    def generate_tokenised_ssv(self):

        mmcif_dict = MMCIF2Dict(f'{self.test_cif_dir}/{self.test_1OJ6_4chains_cif}')
        pdbid = self.test_1V5H_cif
        cif_pdfs_per_chain = cif_parser.parse_cif(mmcif_dict=mmcif_dict, pdb_id=pdbid)
        # cif_pdfs_per_chain = cif_pdfs_per_chain[0]
        cif_pdfs_per_chain = tk._remove_all_hydrogen_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._make_new_bckbone_or_sdchain_col(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._only_keep_chains_of_polypeptide(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._only_keep_chains_with_enuf_bckbone_atoms(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._select_chains_to_use(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)

        cif_pdfs_per_chain = tk._assign_backbone_index_to_all_residue_rows(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._enumerate_atoms_and_residues(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        cif_pdfs_per_chain = tk._assign_mean_corrected_coordinates(pdfs=cif_pdfs_per_chain, pdb_id=pdbid)
        pdf_target = cif_pdfs_per_chain[0]
        dh.write_tokenised_cif_to_ssv(pdf=pdf_target,
                                      path_dst_dir=self.test_tokenised_dir, pdb_id=pdbid)

    def test_parse_cif(self):
        parsed_pdfs = cif_parser.parse_cif(pdb_id=self.test_1OJ6_4chains_cif, relpath_cifs_dir=self.test_cif_dir)
        pass

    def test_parse_tokenise_and_write_cif_to_flatfile(self):
        pdf = tk.parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir=self.test_cif_dir,
                                                       relpath_toknsd_ssv_dir=self.test_tokenised_dir,
                                                       relpath_pdblst=None,
                                                       pdb_ids=self.test_1V5H_ssv[:-4], all_pretokenised_ssvs=None)
        # pdf.to_csv(path_or_buf='test_data/tokenised/test_1V5H.ssv', sep=' ', index=False)
        self.assertEqual(18, len(pdf.columns))

    def test_parse_tokenise_and_write_cif_to_flatfile__4_chains(self):
        list_of_pdfs = tk.parse_tokenise_write_cifs_to_flatfile(relpath_cif_dir=self.test_cif_dir,
                                                                relpath_toknsd_ssv_dir=self.test_tokenised_dir,
                                                                relpath_pdblst=None,
                                                                pdb_id=self.test_1OJ6_4chains_cif)
        for pdf in list_of_pdfs:
            self.assertEqual(18, len(pdf.columns))

    def test_datatypes_after_casting(self):
        pdf_merged = pd.read_csv(f'{self.test_joined_dir}/{self.test_1V5H_ssv}', sep=' ')
        pdf_merged = cif_parser._cast_number_strings_to_numeric_types(pdf_merged)
        pdf_merged = cif_parser._cast_objects_to_stringdtype(pdf_merged)
        expected_int64_dtype = pd.Int64Dtype()
        self.assertEqual(pdf_merged['S_seq_id'].dtype, expected_int64_dtype)
        self.assertEqual(pdf_merged['A_label_seq_id'].dtype, expected_int64_dtype)
        self.assertEqual(pdf_merged['S_pdb_seq_num'].dtype, expected_int64_dtype)
        self.assertEqual(pdf_merged['A_id'].dtype, expected_int64_dtype)
        self.assertEqual(pdf_merged['S_mon_id'].dtype, 'string')
        self.assertEqual(pdf_merged['A_label_comp_id'].dtype, 'string')
        self.assertEqual(pdf_merged['A_label_asym_id'].dtype, 'string')
        self.assertEqual(pdf_merged['S_asym_id'].dtype, 'string')
        self.assertEqual(pdf_merged['A_Cartn_x'].dtype, 'float64')
        self.assertEqual(pdf_merged['A_Cartn_y'].dtype, 'float64')
        self.assertEqual(pdf_merged['A_Cartn_z'].dtype, 'float64')
        self.assertEqual(pdf_merged['A_occupancy'].dtype, 'float64')

    def test__split_up_into_different_chains(self):
        mmcif_dict = MMCIF2Dict(f'{self.test_cif_dir}/{self.test_1OJ6_4chains_cif}')
        polyseq_pdf = cif_parser._extract_fields_from_poly_seq(mmcif_dict)
        atomsite_pdf = cif_parser._extract_fields_from_atom_site(mmcif_dict)
        atomsite_pdf = cif_parser._remove_hetatm_rows(atomsite_pdf)
        chains_pdfs = cif_parser._split_up_by_chain(atomsite_pdf, polyseq_pdf)
        for chain_pdfs in chains_pdfs:
            pass

    def test_some_pd_func(self):
        pdf_target = pd.read_csv(f'{self.test_tokenised_dir}/{self.test_1V5H_ssv_small}', sep=' ')
        pdf_target_deduped = (pdf_target
                              .drop_duplicates(subset=CIF.S_seq_id.value, keep='first')
                              .reset_index(drop=True))
        pass