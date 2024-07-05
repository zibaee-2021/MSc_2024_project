"""
The classification is hierarchical. Hence, if Column 6 (`S35`) for 2 proteins is the same they have at least 35%
sequence identity, if those 2 proteins have same `S60` then they also have at least 60 % identity.
But if you see 2 proteins that have same  `S60`, but different `S35` then they do not have at least 60% sequence
identity. (I need to confirm that I have understood this completely correctly though).

------------------------------------------------------------------------------------------------------------------
Copy-pasted from `README-cath-list-file-format.txt`

Column 1:  CATH domain name (seven characters)
Column 2:  Class number
Column 3:  Architecture number
Column 4:  Topology number
Column 5:  Homologous superfamily number
Column 6:  S35 sequence cluster number
Column 7:  S60 sequence cluster number
Column 8:  S95 sequence cluster number
Column 9:  S100 sequence cluster number
Column 10: S100 sequence count number
Column 11: Domain length
Column 12: Structure resolution (Angstroms)
           (999.000 for NMR structures and 1000.000 for obsolete PDB entries)

CATH Domain Names
-----------------
The domain names have seven characters (e.g. 1oaiA00).

CHARACTERS 1-4: PDB Code
The first 4 characters determine the PDB code e.g. 1oai

CHARACTER 5: Chain Character
This determines which PDB chain is represented.
Chain characters of zero ('0') indicate that the PDB file has no chain field.

CHARACTER 6-7: Domain Number
The domain number is a 2-figure, zero-padded number (e.g. '01', '02' ... '10', '11', '12'). Where the domain number is
a double ZERO ('00') this indicates that the domain is a whole PDB chain with no domain chopping.

------------------------------------------------------------------------------------------------------------------
Copy-pasted from header of `cath-domain-list.txt`

# FILE DESCRIPTION:
# Contains all classified protein domains in CATH
# for class 1 (mainly alpha), class 2 (mainly beta),
# class 3 (alpha and beta) and class 4 (few secondary structures).

"""
import pandas as pd
import os


if __name__ == '__main__':
    print(os.getcwd())
    regex_one_or_more_whitespace_chars = r'\s+'
    pdf = pd.read_csv('../data/dataset/big_files_to_git_ignore/cath-domain-list.txt',
                      skiprows=16,
                      sep=regex_one_or_more_whitespace_chars,
                      names=['DomainID',
                             'Class',
                             'Architecture',
                             'Topology',
                             'HomologousSF',
                             'S35',
                             'S60',
                             'S95',
                             'S100',
                             'S100_Count',
                             'Domain_len',
                             'Angstroms'])  # 500238 proteins (12 columns)
    nmr_pdf = pdf.loc[pdf['Angstroms'] == 999.0]  # 9357 proteins
    obsolete_pdf = pdf.loc[pdf['Angstroms'] == 1000.0]  # 28191 proteins

    # FILTER OUT NMR and 'obsolete' entries (i.e. Angstroms = 999 or 1000)
    pdf_xray = pdf.loc[pdf['Angstroms'] < 999]  # 462690 proteins

    # FILTER OUT LOWER RESOLUTION RECORDS
    pdf_xray = pdf_xray.loc[pdf['Angstroms'] < 4]  # 461083 proteins

    # FILTER IN CLASS 1, 2 or 3 only
    pdf_xray = pdf_xray.loc[pdf_xray['Class'].isin([1, 2, 3])]  # 453364 proteins

    # FILTER OUT SEQUENCE IDENTITY COLUMNS:
    pdf_xray = pdf_xray[['DomainID',
                         'Architecture',
                         'Topology',
                         'HomologousSF',
                         'Domain_len',
                         'Angstroms']]

    # FILTER IN ROWS WITH UNIQUE ['Architecture', 'Topology', 'HomologousSF'] VALUES:
    pdf_xray = pdf_xray[~pdf_xray[['Architecture', 'Topology', 'HomologousSF']].duplicated(keep=False)]  # 550 proteins
    pdf_xray.to_csv(path_or_buf=f'../data/dataset/cath_single_domain_550prots.csv', index=False)

