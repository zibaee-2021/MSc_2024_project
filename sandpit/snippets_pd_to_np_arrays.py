# Removed from tokeniser.py

# # Amino acid indices
# aa_index = np.asarray(pdf_cif[CIF.S_seq_id.value].tolist(), dtype=np.uint16)
# # Atom indices
# pdf_cif[CIF.A_id.value].fillna(0, inplace=True)
# atom_index = np.asarray(pdf_cif[CIF.A_id.value].tolist(), dtype=np.uint16)

# alpha_carbon_indices = np.where(pdf_cif[CIF.A_label_atom_id.value] == 'CA',
#                                 pdf_cif[CIF.A_id.value], np.nan)
# alpha_carbon_indices = alpha_carbon_indices[~np.isnan(alpha_carbon_indices)]
# alpha_carbon_indices = np.asarray(alpha_carbon_indices, dtype=np.uint16)