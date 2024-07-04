import tokeniser as tk

# TRAINING: input is amino acid sequence --> output should be xyz coords for atoms that are seen, else nan.

if __name__ == '__main__':

    tk.write_tokenised_cif_to_csv(pdb_ids=['1MWD', '4hb1'])

