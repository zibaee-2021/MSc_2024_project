import os
import glob
from enum import Enum
import tokeniser as tk

# TRAINING: input is amino acid sequence --> output should be xyz coords for atoms that are seen, else nan.


class PDBidParticularProteins(Enum):
    Myoglobin_pig = '1mwd'  # WT deoxy myoglobin
    Serine_Protease_Human = '4Q7Z' # 'Q6UWY2' X-ray 1.40 Ang aa34-283
    Low_resolution_structure = ''  # resolution > 4 angstroms ??
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)


if __name__ == '__main__':
    cifs = glob.glob(os.path.join('../data/cifs', '*.cif'))
    path_cifs = [cif for cif in cifs if os.path.isfile(cif)]

    pdb_ids = []
    for path_cif in path_cifs:
        cif = os.path.basename(path_cif)
        cif = os.path.splitext(cif)[0]
        pdb_ids.append(cif.rstrip('.cif'))

    tk.write_tokenised_cif_to_csv(pdb_ids)

