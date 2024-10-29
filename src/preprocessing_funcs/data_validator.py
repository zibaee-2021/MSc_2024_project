"""
Functions for performing validation checks on data read from cif files, parsed, tokenised and prepared for training
and validation datasets.
"""

import pandas as pd
from src.preprocessing_funcs.cif_parser import CIF


def check_protein_and_atom_numbering_of_parsed_tokenised_cif_ssv(pdb: dict):
    """

    :param pdb:
    :return:
    """
    pdb_id, pdf_to_profile = pdb.popitem()
    # pdf_to_profile = pdb[pdb_id]
    print(f'Profiling dataframe from flatfile of parsed tokenised cif. \nFor pdb id {pdb_id}.')
    print(f'Column names: {pdf_to_profile.columns.tolist()}. \nShape: {pdf_to_profile.shape}.')
    rows_with_na = pdf_to_profile.isna().any(axis=1).sum()
    print(f'Number of rows with missing values = {rows_with_na}')
    if rows_with_na != 0:
        missing_values = pdf_to_profile.isna().stack()
        missing_locations = missing_values[missing_values].index.tolist()
        print("Missing values found at:")
        for row, col in missing_locations:
            print(f"Row: {row}, Column: {col}")
    num_of_chains = pdf_to_profile[CIF.S_asym_id.value].nunique()
    chains = pdf_to_profile[CIF.S_asym_id.value].unique().tolist()
    print(f'cif with pdb id={pdb_id} has {num_of_chains} chains. \nThey are {chains}.')

    for chain in chains:
        print('CHAIN {chain}')
        one_chain_pdf = pdf_to_profile[pdf_to_profile[CIF.S_asym_id.value] == chain]

        first_value = one_chain_pdf.iloc[0][CIF.S_seq_id.value]
        print(f'First row value: {first_value} in column: {CIF.S_seq_id.value}')
        last_value = one_chain_pdf.iloc[-1][CIF.S_seq_id.value]
        print(f'Last row value: {last_value} in column: {CIF.S_seq_id.value}')

        first_value = one_chain_pdf.iloc[0][CIF.A_id.value]
        print(f'First row value: {first_value} in column: {CIF.A_id.value}')
        last_value = one_chain_pdf.iloc[-1][CIF.A_id.value]
        print(f'Last row value: {last_value} in column: {CIF.A_id.value}')

        is_incremental = (one_chain_pdf[CIF.S_seq_id.value].diff().iloc[1:] == 1).all()
        increment = '' if is_incremental else 'not '
        print(f'Numbers in {CIF.S_seq_id.value} do {increment}increment by one as you go down each row.')

        is_increasing = (one_chain_pdf[CIF.A_id.value].diff().iloc[1:] > 0).all()
        increase = '' if is_increasing else 'not '
        print(f'Numbers in {CIF.A_id.value} do {increase}always increase as you go down the rows.')

