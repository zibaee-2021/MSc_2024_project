MSc student project 
-
---
#### SHELL SCRIPT LAUNCHERS:

You can run the Python scripts via shell scripts in `launchers` directory in the top level of the project folder:  

`sh launchers/tokenise_embed_train.sh`  
This calls `src/diffusion/tokenise_embed_train.py`  

Alternatively, for demonstration, each part can be run separately:  
- `sh launchers/_1_run_tokeniser.sh`  
- `sh launchers/_2_build_embeddings.sh` 
- `sh launchers/load_datasets`
- `sh launchers/_3_train_model.sh`

> Note: Running any of these launcher shell scripts will automatically activate the conda env (`diffSock` on `joe-desktop`).
> However, if you try to run one of the shell scripts within an already-activated conda env, it seems to cause 
> some form of conflict and fails to execute it, throwing error messages about being unable to find required 
> libraries (like NumPy and Pandas).
---
#### Specifying which proteins to train model on:
- Before running anything, you must decide which proteins will be used for training the model. 
- Specifying which `PDBids`/`PDBid_chains` to use is done via `tokeniser.py` script.
- Once `tokeniser.py` has run, the remaining scripts (embedding and model training) will know implicitly which PDBids 
to work with via reading in .ssv files written to `src/diffusion/diff_data/tokenised` by `tokeniser.py`, so it is not 
necessary to indicate which PDBids to those scripts as well, only to `tokeniser.py`.

---
#### TOKENISER

`src/preprocessing_funcs/tokeniser.py`

- `tokeniser.py` builds a list of `PDBids` and/or `PDBid_chains` according to one or more of the 3 following paths 
passed to it. (All of these are optional, as default arguments are also provided.)

1. User-specified directory path from where pre-downloaded `mmCIF` files will be read.  
The default is `mmCIF` directory at `src/diffusion/diff_data`. Whichever `.cif` files are found there will be parsed and 
tokenised (unless it finds corresponding `.ssv` files already in `tokenised` directory at `src/diffusion/diff_data`, 
indicating which, if any, have already been tokenised). 

2. User-specified file path to any existing `.lst` file of `PDBids` and/or `PDBid_chains`.   
A number of these `.lst` files is located in `PDBid_list` at `src/diffusion/diff_data`.   
The default is `None`.  
However, you can also pass the PDBids `.lst` file name at the command line.  
Currently, `pdbchains_565` is included at the command-line in both shell scripts that call `tokeniser.py`.  
As a result, the tokeniser searches for `src/diffusion/diff_data/PDBid_list/pdbchains_565.lst`.  
If it finds this file, it will read in the PDBids from it.

3. User-specified string of `PDBid` and/or `PDBid_chain`; or a Python list of `PDBid` and/or `PDBid_chain` strings.    
(Either a single string or Python list of strings can be passed to `pdb_ids` argument.)
The default is `None`.  

If by the end of this, the list of `PDBids` is still empty, this will be reported (in print statement) and is hard-coded 
to use `src/diffusion/diff_data/PDBid_list/pdbchains_565.lst` (for simpler testing/demo experience).

> The `tokenise_embed_train.py` script is currently set up to first empty the `mmCIF` and `tokenised` directories, 
in order to remove any pre-downloaded `mmCIF` files and remove any pre-tokenised `.ssv` files, so that they are all 
downloaded/generated in *this* run.  
If you don't want this to happen but instead to re-use the files in those directories, then you must comment these 
out manually on lines 11 & 12.  
Conversely, the `_1_run_tokeniser.sh` script will call `tokeniser.py` directly via its `if __name__ == __main__` 
wherein it is currently set up to *not* empty those two directories.  
So this time you would need to <u>un</u>comment them on lines 616 and/or 619 if you want the mmCIFs to be downloaded 
from the API in *this* run and/or for their tokenisation to be performed in *this* run, respectively.

  
---
#### PROTEIN LANGUAGE MODEL EMBEDDER

`src/pLM/plm_embedder.py`  

The choice of which `PDBid_chains` to build protein language model (PLM) embeddings for is implicitly indicated by the tokeniser.  
`tokeniser.py` (which must be run *before* the `plm_embedder.py`) writes tokenised data to `.ssv` files in `src/diffusion/diff_data/tokenised`.  
`plm_embedder.py` reads these `.ssv` files in and extracts the `PDBid_chain` and amino acid sequence directly from each one.  
(In this way, it is guaranteed that in the subsequent section, the neural networks which train on atomic-level data 
are aligned with those that train on amino acid sequence-level data.)  

---
#### MODEL TRAINING

`src/diffusion/pytorch_protfold_allatomclustrain_singlegpu.py` 
`src/diffusion/nndef_protfold_atompyt2.py`

As with the pLM embedder, the choice of which `PDBid_chains` to train on is implicitly indicated by the tokeniser.  
`tokeniser.py` (which must be run *before* the training script) writes tokenised data to `.ssv` files in 
`src/diffusion/diff_data/tokenised`.  
`pytorch_protfold_allatomclustrain_singlegpu.py` reads these `.ssv` files in and prepares two datasets via 
`src/diffusion/dataset_loader.py`. 
It is also necessary that pLM embeddings, generated by `plm_embedder.py`, will already have been run *before* starting
the training script.  
Without these embeddings, saved to `.pt` files in `src/diffusion/diff_data/emb`, the training script will fail to execute.


> The `tokenise_embed_train.py` script is currently set up to first empty the `emb` directory, in order to remove 
any pre-built embedding `.pt` files, so that they are all generated in *this* run.  
If you don't want this to happen but instead to re-use the files in those directories in the subsequent model training, 
then you must comment this out manually on line 16.  
Conversely, the `_2_build_embeddings.sh` script will call `plm_embedder.py` directly via its `if __name__ == __main__` 
wherein it is currently set up to *not* empty that directory.  
So this time you would need to <u>un</u>comment it on line 106 if you want the embeddings to be generated in *this* run.

---
#### ENVIRONMENT, DEPENDENCIES & HARDWARE:

All code was run in miniconda3 environment.

| Package            | version (macOS 14.7.2 Intel x86_64)  | version (Rocky Linux 9.5, x86-64)       |
|--------------------|--------------------------------------|-----------------------------------------|
| <code>conda</code> | `24.11.3`                            | `24.11.0`                               |
| <code>python</code>   | `3.12.3`                             | `3.12.2`                                |
| `numpy`            | `1.26.4`                             | `2.0.1`                                 |
| `pandas`           | `2.2.3`                              | `2.2.3`                                 |
| `biopython`        | `1.84`                               | `1.84`                                  |
| `requests`         | `2.32.3`                             | `2.32.3`                                |
| `yaml`             | `0.2.5`                              | `0.2.5`                                 |
| `matplotlib`       | `3.10.0`                             | `3.9.2`                                 |
| `transformers`     | `4.48.1 pip install transformers -U` | `4.46.3 conda-forge`                    |
| `pytorch`          | NA                                   | `2.5.1   py3.12_cuda12.1_cudnn9.1.0_0 ` |
| `torchvision`      | NA                                   | `0.20.1`                                |
| `pytorch-cuda`     | NA                                   | `12.1`                                  |
| `einops`           | NA                                   | `0.8.0`                                 |

`cuda` packages (Rocky Linux only):

| Package         | Version    | Build | Channel |
|-----------------|------------|-------|---------|
| `cuda-cudart`     | `12.1.105`  | `0`     | nvidia  |
| `cuda-cupti`      | `12.1.105`  | `0`     | nvidia  |
| `cuda-libraries`  | `12.1.0`    | `0`     | nvidia  |
| `cuda-nvrtc`      | `12.1.105`  | `0`     | nvidia  |
| `cuda-nvtx`       | `12.1.105`  | `0`     | nvidia  |
| `cuda-opencl`     | `12.6.77`   | `0`     | nvidia  |
| `cuda-runtime`    | `12.1.0`    | `0`     | nvidia  |
| `cuda-version`    | `12.6`      | `3`     | nvidia  |

GPU Rocky Linux = NVIDIA GeForce GTX 1080 Ti

---
##### Summary of Python scripts and other files:

- `src/preprocessing_funcs` contains 6 Python scripts and 2 text files that are purely documentation.
  - The following 4 scripts are needed for generating a suitable list of proteins and tokenising their `mmCIF` data.
    - `single_dom_builder.py` builds a dataset of single domain protein using CATH, and downloads their `mmCIF` files.
    - `tokeniser.py` tokenises residues of protein sequence & structural data of corresponding atoms of 1 protein chain, per PDB id from given list of ids. 
    - `cif_parser.py` parses `mmCIF` files in preparation for tokenisation by `tokeniser.py`.
    - `api_caller.py` HTTP client to download cif files from `files.rcsb.org` or FASTA sequences from `www.uniprot.org`.
  
  - The following 2 scripts are "helper functions", but not currently used:
    - `faa_to_atoms.py` helper function to convert a given amino acid sequence to their corresponding atomic sequence.
    - `FASTA_reader.py` helper function to read amino acid sequence of protein given its Uniprot Accession number.

To run `single_domain_builder.py`, the `if __name__ == '__main__':` is currently set up to run the main function 
`parse_single_dom_prots_and_write_csv()` which reads the locally downloaded `cath-domain-list.txt` 
(which is excluded from git tracking due to size, but should be found on local machine in 
`data/dataset/big_files_to_git_ignore/CATH`.) This function saves the parsed data for a selection of the 
single-domain proteins in which a unique structural and functional category is represented only once and which satisfy
certain constraints on the structural data - resulting in a list of only 573 single domain proteins.   
Hence if you want to see this run, simply use `python3 single_dom_builder.py`.

----
##### Layout of functions in src/preprocessing_funcs, src/pLM & data_layer:

The format these scripts follows is for there to be 1 to 3 public functions in one script. 
The public functions are positioned from the bottom up of the script and in the order in which they are called. 
Each public function may call private functions (indicated by underscore prefix) and these are also positioned from 
bottom up in the order in which they are called.

For example: 
```python
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
#### COMPILING LIST OF SINGLE-DOMAIN PROTEINS VIA `CATH` WEBSERVER:

Performed by `single_dom_builder.py`

The entire list of domain ids from the latest official version from the file `cath-domain-list.txt` was manually downloaded from CATH webserver at:
`http://download.cathdb.info/cath/releases/all-releases/latest_release/cath-classification-data/cath-domain-list.txt` 
and saved to `data/dataset/big_files_to_git_ignore/CATH/cath-domain-list.txt` (which I haven't added to git but is on `joe-desktop`).
Starting then from 500,238 proteins (and 12 columns of data fields), the dataset is filter by process of elimination:
- removing records for NMR structures, obsolete structures & low resolution X-ray structures
- keep only proteins of classes 1, 2 or 3
- remove data fields that are related to sequence identity 
- store PDBids in new column, which helps filter in those with only one domain
- filter in PDBs with unique 'Architecture', 'Topology' and 'HomologousSF' values.
  - results in reducing number of PDBs from 28,496 proteins to only 573, but this provides a diverse dataset and serves 
   well as a 'dummy' dataset. 
- The `PDB_Id`, `DomainID`, `Architecture` , `Topology`, `HomologousSF`, `Domain_len`, `Angstroms` fields of this 573 
single-domain diverse set of proteins PDBids is saved to  `data/dataset/big_files_to_git_ignore/CATH/SD_573_CIFs.csv`
  (Ref: Orengo et al. 1997 'CATH — a hierarchic classification of protein domain structures' Structure 1997, Vol 5 No 8, pp1093-1108)

----
#### Data handling 'helper' functions:

`data_layer/data_handler.py` contains about 30 functions that focus on reading, writing and moving data.
(The original directory structure design was to store all data in the top-level `data` folder. However in the current
format I have most of the data like `mmCIF` files, tokenisation, pLM embeddings in `src/diffusion/diff_data`, which was
done for temporary reasons, but has remained there for now. (I will move them out to the top-level data directory in future).)

---
END