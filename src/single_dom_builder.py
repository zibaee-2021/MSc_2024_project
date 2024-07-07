"""
Use CATH protein domain web server to extract single domain proteins with high resolution X-ray data.
I downloaded cath-domain-list.txt locally first (in `../data/dataset/big_files_to_git_ignore`), then parsed it using
Pandas.

Some explanation of Columns 6 to 10, as I understand it:

The classification is hierarchical. Hence, if Column 6 (`S35`) for 2 proteins is the same they have at least 35%
sequence identity, if those 2 proteins have same `S60` then they also have at least 60 % identity.
But if you see 2 proteins that have same  `S60`, but different `S35` then they do not have at least 60% sequence
identity. (I need to confirm that I have understood this completely correctly though).

------------------------------------------------------------------------------------------------------------------
The following is copy-pasted from `README-cath-list-file-format.txt`

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
The following is copy-pasted from header of `cath-domain-list.txt`

# FILE DESCRIPTION:
# Contains all classified protein domains in CATH
# for class 1 (mainly alpha), class 2 (mainly beta), class 3 (alpha and beta) and class 4 (few secondary structures).

"""
import os
from enum import Enum
import pandas as pd
import requests


global non_200_count


class Cols(Enum):
    DomainID = 'DomainID'
    C = 'Class'
    A = 'Architecture'
    T = 'Topology'
    H = 'HomologousSF'
    Domain_len = 'Domain_len'
    Angstroms = 'Angstroms'

    PDB_ID = 'PDB_Id'


def parse_single_domain_prots_and_write_to_csv(path_cath_list: str, path_single_dom_prots: str):
    regex_one_or_more_whitespace_chars = r'\s+'
    pdf = pd.read_csv(path_cath_list,
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
    # The following two dataframe are not used. I just wanted to see how many of these other categories there are:
    nmr_pdf = pdf.loc[pdf[Cols.Angstroms.value] == 999.0]  # 9357 proteins
    obsolete_pdf = pdf.loc[pdf[Cols.Angstroms.value] == 1000.0]  # 28191 proteins

    # FILTER OUT NMR and 'obsolete' entries (i.e. Angstroms = 999 or 1000)
    pdf_xray = pdf.loc[pdf[Cols.Angstroms.value] < 999]  # 462690 proteins

    # FILTER OUT LOWER RESOLUTION RECORDS
    pdf_xray = pdf_xray.loc[pdf[Cols.Angstroms.value] < 4]  # 461083 proteins

    # FILTER IN CLASS 1, 2 or 3 only
    pdf_xray = pdf_xray.loc[pdf_xray[Cols.C.value].isin([1, 2, 3])]  # 453364 proteins

    # FILTER OUT SEQUENCE IDENTITY COLUMNS:
    pdf_xray = pdf_xray[[Cols.DomainID.value,
                         Cols.A.value,
                         Cols.T.value,
                         Cols.H.value,
                         Cols.Domain_len.value,
                         Cols.Angstroms.value]]

    # Dupes is not used. I just wanted to confirm for myself that DomainID column has unique values:
    dupes = pdf_xray[pdf_xray[Cols.DomainID.value].duplicated(keep=False)]

    # FILTER TO SINGLE-DOMAIN PROTEINS:

    # Add new column with just PDB ids (first 4 characters of `DomainID`) and make it the first column:
    pdf_xray[Cols.PDB_ID.value] = pdf_xray[Cols.DomainID.value].str[:4]
    first_col = pdf_xray.pop(Cols.PDB_ID.value)
    pdf_xray.insert(0, Cols.PDB_ID.value, first_col)

    # This PDB id will be repeated for multidomain proteins (row represents a domain and a pdb structure record)
    domain_counts = pdf_xray[Cols.PDB_ID.value].value_counts().sort_values()
    # I just wanted to scroll through this very large dataframe, which is easier with a csv:
    # domain_counts.to_csv('../data/dataset/domain_counts.csv')

    # Hence, keep only those protein records where the PDB occupies only one row:
    single_domain_prots = domain_counts[domain_counts == 1].index.tolist()
    pdf_single_dom_prots = pdf_xray[pdf_xray[Cols.PDB_ID.value].isin(single_domain_prots)]  # 28496 proteins

    # FILTER IN ROWS WITH UNIQUE ['Architecture', 'Topology', 'HomologousSF'] VALUES.
    # This is so that each unique structural and functional category is represented only once.
    # I can afford to do this because I only need a "few hundred" single domain proteins:
    pdf_single_dom_prots = pdf_single_dom_prots[~pdf_single_dom_prots[[Cols.A.value,
                                                                       Cols.T.value,
                                                                       Cols.H.value]].duplicated(keep=False)]  # 573
    # proteins
    pdf_single_dom_prots.to_csv(path_single_dom_prots, index=False)


def fetch_mmcif_from_pdb_api_and_write_locally(pdb_id):

    global non_200_count
    url = f'https://files.rcsb.org/download/{pdb_id}.cif'
    response = requests.get(url)
    response.raise_for_status()
    code = response.status_code
    if code != 200:
        non_200_count += 1
        print(f'Response status code for {pdb_id} is {code}, hence could not read the pdb for this id.')

    mmcif_file = f'../data/cifs/{pdb_id}.cif'
    with open(mmcif_file, 'w') as file:
        file.write(response.text)


if __name__ == '__main__':
    path_cath_domain_list = '../data/dataset/big_files_to_git_ignore/cath-domain-list.txt'
    path_single_dom_prots = '../data/dataset/cath_573_single_domain_prots.csv'
    parse_single_domain_prots_and_write_to_csv(path_cath_list=path_cath_domain_list,
                                               path_single_dom_prots=path_single_dom_prots)
    non_200_count = 0

    if os.path.exists(path_single_dom_prots):
        pdf_573prots = pd.read_csv(path_single_dom_prots)

    print(f'number of domain ids = {pdf_573prots.shape}')
    pdb_ids = pdf_573prots[Cols.PDB_ID.value].tolist()

    for pdb_id in pdb_ids:
        fetch_mmcif_from_pdb_api_and_write_locally(pdb_id)
    print(f'Total number of non-200 status codes is {non_200_count}')
    # This seems to only return 465 proteins out the 550 domain ids

    # programmatically count cifs in `../data/cifs`
    import glob
    files = glob.glob(os.path.join('../data/cifs', '*.cif'))
    files = [file for file in files if os.path.isfile(file)]
    print(f'There are {len(files)} cifs in that dir')
    # There are indeed 464 cifs in there

