MSc student project 

#### ENVIRONMENT, DEPENDENCIES & HARDWARE:
All code was run in a conda environment.
My installed packages on MacOS 14.7.2 Intel (CPU is suitable for tokenisation):
- Conda 24.11.2 (miniconda3)
- Python 3.12.3
- NumPy 1.26.4
- Pandas 2.2.3
- Biopython 1.84
- Requests 2.32.3
- Yaml 0.2.5
- Matplotlib 3.10.0

Installed packages on Rocky Linux ... 
- Conda 24.11.2 (miniconda3)
- Python 3.12.3
- NumPy 2.2... 
- Pandas 2.2.3
- Biopython 1.84
- Requests 2.32.3
- Yaml 0.2.5
- Matplotlib 3.10.0

- PyTorch 2.3... 
- torchvision 0.18.1
- transformers 4.46.2 (HuggingFace package)
- 
Installed packages on HPC ... 



Summary of Python scripts and other files:
- `src/preprocessing_funcs` contains 6 Python scripts and 2 text files that are purely documentation.
  - The following 4 scripts are needed for generating a suitable list of proteins and tokenising their mmCIF data.
    - `single_dom_builder.py` builds a dataset of single domain protein using CATH, and downloads their mmCIF files.
    - `tokeniser.py` tokenises residues of protein sequence & structural data of corresponding atoms of 1 protein chain, per PDB id from given list of ids. 
    - `cif_parser.py` parses mmCIF files in preparation for tokenisation by `tokeniser.py`.
    - `api_caller.py` HTTP client to download cif files from `files.rcsb.org` or FASTA sequences from `www.uniprot.org`.
  
  - The following 2 scripts are "helper functions", but not currently used:
    - `faa_to_atoms.py` helper function to convert a given amino acid sequence to their corresponding atomic sequence.
    - `FASTA_reader.py` helper function to read amino acid sequence of protein given its Uniprot Accession number.

To run `single_domain_builder.py`, the `if __name__ == '__main__':` is currently set up to run the main function 
`parse_single_dom_prots_and_write_csv()` which reads the locally downloaded `cath-domain-list.txt` 
(which is excluded from git tracking due to size, but should be found on local machine in 
`data/dataset/big_files_to_git_ignore/CATH`.) This function saves the parsed data for a selection of the 
single domain proteins in which a unique structural and functional category is represented only once and which satisfy
certain constraints on the structural data - resulting in a list of only 573 single domain proteins.   
Hence if you want to see this run, simply use `python3 single_dom_builder.py`.

----
#### LAYOUT OF PYTHON SCRIPTS FOR PREPARATION OF TRAINING DATASET: 
Location of scripts: `src/preprocessing_funcs` and `data_layer`.

The typical format of these scripts is for there to be 1 to 3 public functions in one script. 
The public functions are positioned from the bottom up of the script and in the order in which they are called. 
Each public function may call private functions and these are positioned from bottom up in the order in which they 
are called.

For example: 
```
def public_func2(val1):
    return some_operation / val1

def _private_func2(res)
    return some_operation * res
    
def _private_func1()
    return some_operation

def public_func1():
    result = _private_func1()
    return _private_func2(result)

if __name__ == '__main__':
    value1 = public_func1()
    public_func2(value1)
```
----
#### 1. GENERATE DATASET OF SINGLE-DOMAIN PROTEINS VIA CATH WEBSITE:

Performed by `single_dom_builder.py`

The entire list of domain ids from the latest official version from the file:

`cath-domain-list.txt` is manually downloaded from CATH webserver at:
`http://download.cathdb.info/cath/releases/all-releases/latest_release/cath-classification-data/cath-domain-list.txt` 
and downloaded to `data/dataset/big_files_to_git_ignore/CATH/cath-domain-list.txt` (which I don't add to git).
Starting then from 500238 proteins (and 12 columns of data fields), the dataset is filter by:
- removing records for NMR structures, obsolete structures & low resolution X-ray structures
- keep only proteins of classes 1, 2 or 3
- remove data fields that are related to sequence identity 
- store PDB ids in new column, which helps filter in those with only one domain
- filter in PDBs with unique 'Architecture', 'Topology' and 'HomologousSF' values.
  - results in reducing number of PDBs from 28496 proteins to only 573, but this provides a diverse dataset and serves 
   well as a 'dummy' dataset. 
- The `PDB_Id`, `DomainID`, `Architecture` , `Topology`, `HomologousSF`, `Domain_len`, `Angstroms` fields of this 573 
single-domain diverse set of proteins PDB ids is saved to  `data/dataset/big_files_to_git_ignore/CATH/SD_573_CIFs.csv`
----
#### 2. GENERATE PROTEIN LANGUAGE MODEL EMBEDDINGS FOR EACH PROTEIN:


----
#### 3. READ, PARSE AND TOKENISE mmCIF FILES:

Performed by `tokeniser.py` (which imports and uses `cif_parser.py`, `api_caller.py`, `data_layer/data_handler.py`).

This can be run as part of the training protocol, or separately (i.e. via its own `if __name__ == '__main__':`). 
Running it separately beforehand, whereby each PDB's raw mmCIF data is saved to `src/diffusion/diff_data/mmCIF` and 
the tokenised mmCIF data is saved in ssv files to `src/diffusion/diff_data/tokenised` was deemed preferable as, 
although it requires more disk memory, it reduces risk of unnecessary repetition, particularly during development as 
the quality and potential problems with mmCIF data were being discovered and code updated to handle these accordingly. 

The process of reading in, parsing and tokenising protein structure data of choice is initiated by passing the 
PDB ids of those proteins to `tokeniser.parse_tokenise_write_cifs_to_flatfile()`.

`mmCIF` files are searched for first in `src/diffusion/diff_data/mmCIF` in case already downloaded.
If not, they will be automatically downloaded directly from `https://files.rcsb.org/download/{pdb_id}.cif` via 
`api_caller.py` and saved to `src/diffusion/diff_data/mmCIF`.

Once read in and converted to a Python dict via Biopython library, `cif_parser.parse_cif()` extracts the required 
atom-related subfields from `_atom_site` field and the required protein sequence-related subfields from 
`_pdbx_poly_seq_scheme` and stores these in a Pandas dataframe for improved readability and to make available the 
powerful data wrangling API of Pandas dataframes. 

The descriptive names of the private functions called by `cif_parser.parse_cif()` make them self-explanatory. 
Furthermore the mmmCIF fields are heavily documented in the docstring at the top of the script, 
as well as throughout the code. Nonetheless, a summary of the main operations of this script is given:
- extract required data 
- handle low occupancy data 
- cast to numeric data types 
- separate cif data out by chain into a list of dataframes with one PDB-chain combination per dataframe, 
- remove missing data (the option to impute missing data with some value is not used but remains in the code)
- clean and tidy up. 

This list of PDB-chain dataframes are returned to `tokeniser.parse_tokenise_write_cifs_to_flatfile()`
Once again, the descriptive names of the private functions called by `tokeniser.parse_tokenise_write_cifs_to_flatfile()` 
makes them self-explanatory. Nonetheless, a summary of their main operations is given:
- removes hydrogen atoms 
- identifies all backbone atoms
- removes any non-protein chains
- removes chains that lack sufficient backbone atoms
- keep only one chain per protein
- generate numerical representation of residues and atoms (mappings stored in json files in `data/enumeration`)
- store index positions of all "anchor" backbone atoms (i.e. the alpha carbon or other backbone atom)
- calculate mean-corrected atomic coordinate values
- search for any missing data values (should not be any at this stage)
- save this dataframe to ssv files in `src/diffusion/diff_data/tokenised`
- write out list of PDB_chain ids to `.lst` file in `diff_data/PDBid_list/`

----
#### TRAIN NEURAL NETWORK:

