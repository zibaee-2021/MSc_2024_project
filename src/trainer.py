from enum import Enum
import tokeniser as tk

# TRAINING: input is amino acid sequence --> output should be xyz coords for atoms that are seen, else nan.


class PDBid(Enum):
    Myoglobin_pig = '1mwd'  # WT deoxy myoglobin
    Serine_Protease_Human = '4Q7Z' # 'Q6UWY2' X-ray 1.40 Ang aa34-283
    Low_resolution_structure = ''  # resolution > 4 angstroms ??
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)


if __name__ == '__main__':

    # tk.write_tokenised_cif_to_csv(pdb_ids=['1MWD', '4hb1'])
    tk.write_tokenised_cif_to_csv(pdb_ids=['4hb1'])

