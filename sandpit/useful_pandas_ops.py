import pandas as pd

# dups = pdf_merged[pdf_merged.duplicated(subset=[CIF.S_pdb_mon_id.value, CIF.S_auth_mon_id.value], keep=False)]
# comparison = pdf_merged[CIF.S_pdb_mon_id.value] == pdf_merged[CIF.S_auth_mon_id.value]
# identical_count = comparison.sum()
# diff_count = len(comparison) - identical_count
# print(identical_count)
# print(diff_count)
# comp_ = pdf_merged[[CIF.S_pdb_mon_id.value, CIF.S_auth_mon_id.value]]