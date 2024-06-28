import cif_parser as parser
import json
import numpy as np

if __name__ == '__main__':

    pdf_cif = parser.parse_cif(local_cif_file='../data/cifs/4hb1.cif')
    with open('../data/jsons/aas_atoms_enumerated.json', 'r') as json_f:
        aas_atoms_enumerated = json.load(json_f)
    aas_atoms_enumerated = {eval(k): v for k, v in aas_atoms_enumerated.items()}
    print(aas_atoms_enumerated)

    aa_codes = np.asarray(aa_codes, dtype=np.uint8)
    # What DJ generates in `ds_loader.py`:
    # ntcodes = np.asarray(ntcodes, dtype=np.uint8)
    # atomcodes = np.asarray(atomcodes, dtype=np.uint8)
    # bbindices = np.asarray(bbindices, dtype=np.int16)
    # ntindices = np.asarray(ntindices, dtype=np.int16)
    # target is just the name of the cif (i.e. string)
    # target_coords is the

    # ntcodes is a list of the enumerated representation of the sequence. e.g. ACDE gives ntcodes [0, 1, 2, 3]
    # atomcodes is the same as ntcodes but for atoms
    # bbindices is just an index of the atoms (not sure what "bb" stands for)



    # finally output:
    # the tuple `sp = (ntcodes, atomcodes, ntindices, bbindices, target, target_coords)`