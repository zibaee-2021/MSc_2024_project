# Placeholder file for script that will contain all the functions that read/write from/to `data` subdirs, to just make
# the other functions a bit tidier.
import os
from enum import Enum
import glob
import shutil
import re
from typing import List
import json
import pandas as pd
import torch
import yaml
from typing import Tuple
from src.enums import CIF, ColNames
from src.preprocessing_funcs import api_caller as api


class Path(Enum):
    data_pdbid_dir = '../data/PDBid_list'
    sd_573_cifs_dir = '../data/dataset/big_files_to_git_ignore/SD_573_CIFs'
    rp_bigdata_cif_dir = '../data/dataset/big_files_to_git_ignore/SD_573_CIFs'
    rp_bigdata_toknsd_dir = '../data/dataset/big_files_to_git_ignore/tokenised_573'

    enumeration_dir = 'enumeration'
    data_dir = '../data'
    fasta_dir = 'FASTA'
    data_fasta_dir = '../data/FASTA'
    data_tokenised_dir = '../data/tokenised'
    aa_atoms_yaml = '../data/yaml/residues_atoms.yaml'
    data_per_aa_atoms_json = '../data/residues_atoms/per_residue_atoms.json'
    diffdata_cif_dir = '../src/diffusion/diff_data/mmCIF'
    enumeration_h_list = 'enumeration/hydrogens.lst'

    rp_diffdata_sd573_lst = '../diffusion/diff_data/SD_573.lst'
    rp_diffdata_globins10_lst = '../diffusion/diff_data/globins_10.lst'
    rp_diffdata_globin1_lst = '../diffusion/diff_data/globin_1.lst'


class Filename(Enum):
    aa_atoms_no_h = 'residues_atoms_no_hydrogens'
    atoms_no_h = 'unique_atoms_only_no_hydrogens'
    aa = 'residues'


class FileExt(Enum):
    CIF = 'cif'
    dot_CIF = '.cif'
    ssv = 'ssv'
    dot_ssv = '.ssv'
    csv = 'csv'
    tsv = 'tsv'
    dot_lst = '.lst'
    dot_json = '.json'
    dot_txt = '.txt'
    dot_pt = '.pt'


class YamlKey(Enum):
    root = 'ROOT'
    aas = 'AAs'
    atoms_by_aa = 'ATOMS_BY_AA'


def _chdir_to_data_layer():
    """
    Change current working dir to data layer.
    :return: Current working directory *before* changing it to the local dir of data_handler in order to return to it
    at the end of the operation.
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f'Path changed from {cwd} to = {os.getcwd()}. (This is intended to be temporary).')
    return cwd


def _restore_original_working_dir(cwd: str):
    os.chdir(cwd)


def _make_pdbid_uppercase(fpath: str) -> str:
    dir_name, base_name = os.path.split(fpath)
    fname, dot_ext = os.path.splitext(base_name)
    fname = fname.upper()
    return os.path.join(dir_name, f'{fname}{dot_ext}')


def copy_files_over(path_src_dir: str, path_dst_dir: str, file_ext: str) -> int:
    if not os.path.exists(path_src_dir):
        raise ValueError(f"Directory '{path_src_dir}' you're trying to copy from can't be found.")
    if not file_ext:
        raise ValueError('File extension must be given.')
    file_ext = file_ext.removeprefix('.')

    os.makedirs(path_dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(path_src_dir, f'*{file_ext}'))
    file_count = 0
    if not files:
        print(f"No files with extension '{file_ext}' found in '{path_src_dir}'.")
        return file_count
    for fpath in files:
        fname = os.path.basename(fpath)
        dest_path = os.path.join(path_dst_dir, fname)
        dest_path = _make_pdbid_uppercase(dest_path)
        shutil.copy(fpath, dest_path)
        print(f'Copied: {fpath} -> {dest_path}')
    file_count = len(files)
    return file_count


def copy_cifs_from_bigfilefolder_to_diff_data():
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer.
    file_count = copy_files_over(path_src_dir=Path.sd_573_cifs_dir.value,
                                 path_dst_dir=Path.diffdata_cif_dir.value,
                                 file_ext=FileExt.CIF.value)
    print(f"Number of '.{FileExt.CIF.value}' files copied over = {file_count}")
    _restore_original_working_dir(cwd)


def clear_diffdatacif_dir() -> None:
    """
    Note: As we're likely dealing with less than 10,000 files, I am not using Linux via subprocess, which would be:
    ```subprocess.run(['rm', '-rf', f'{directory_path}/*'], check=True, shell=True)```
    """
    for cif_file in os.listdir(Path.diffdata_cif_dir.value):
        cif_path = os.path.join(Path.diffdata_cif_dir.value, cif_file)
        os.unlink(cif_path)


def read_list_of_pdbids_from_text_file(filename: str):
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer.
    path_file = os.path.join(Path.data_pdbid_dir.value, filename)
    with open(path_file, 'r') as f:
        pdb_ids = f.read()
    pdbids = pdb_ids.split()
    _restore_original_working_dir(cwd)
    return pdbids


def read_pdb_lst_from_src_diff_dir(relpath_pdblst: str) -> list:
    """
    Read protein PDB id list from `.lst` file, hence expects one PDBid per line.
    E.g.:
                                    `3C9P
                                     2HL7`
     or PDBid_chain:
                                    `3C9P_A
                                     3C9P_B
                                     2HL7_C`
    :param relpath_pdblst:
    :return:
    """
    try:
        with open(relpath_pdblst, 'r') as pdblst_f:
            pdb_ids = [line.strip() for line in pdblst_f]
    except FileNotFoundError:
        print(f'{pdblst_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return pdb_ids


def get_list_of_pdbids_of_local_single_domain_cifs() -> list:
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    cifs = glob.glob(os.path.join(Path.sd_573_cifs_dir.value, f'*{FileExt.dot_CIF.value}'))
    path_cifs = [cif.upper() for cif in cifs if os.path.isfile(cif)]
    pdb_id_list = []

    for path_cif in path_cifs:
        cif_basename = os.path.basename(path_cif)
        pdbid = os.path.splitext(cif_basename)[0]
        pdb_id_list.append(pdbid)

    _restore_original_working_dir(cwd)
    return pdb_id_list


# def get_list_of_uniprotids_of_locally_downloaded_cifs():


def write_list_to_lst_file(list_to_write: list, path_fname: str) -> None:
    path_fname = path_fname.removesuffix(FileExt.dot_lst.value)
    path_fname = f'{path_fname}{FileExt.dot_lst.value}'
    with open(path_fname, 'w') as f:
        for item in list_to_write:
            f.write(f'{item}\n')


def write_list_to_space_separated_txt_file(list_to_write: list, fname: str) -> None:
    fname = fname.removesuffix(FileExt.dot_txt.value)
    fname = fname + FileExt.dot_txt.value
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    space_sep_str = ' '.join(list_to_write)
    with open(f'{Path.data_dir.value}/{fname}', 'w') as f:
        f.write(space_sep_str)
    _restore_original_working_dir(cwd)


def write_enumerations_json(fname: str, dict_to_write: dict) -> None:
    fname = fname.removesuffix(FileExt.dot_json.value)
    _write_to_json_to_data_dir(fname=f'{Path.enumeration_dir.value}/{fname}{FileExt.dot_json.value}',
                               dict_to_write=dict_to_write)


def read_enumerations_json(fname: str) -> dict:
    fname = fname.removesuffix(FileExt.dot_json.value)
    return _read_json_from_data_dir(fname=f'{Path.enumeration_dir.value}/{fname}{FileExt.dot_json.value}')


def read_enumeration_mappings() -> Tuple[dict, dict, dict]:
    residues_atoms_enumerated = read_enumerations_json(fname=Filename.aa_atoms_no_h.value)
    residues_atoms_enumerated = {eval(k): v for k, v in residues_atoms_enumerated.items()}
    atoms_enumerated = read_enumerations_json(fname=Filename.atoms_no_h.value)
    residues_enumerated = read_enumerations_json(fname=Filename.aa.value)
    return residues_atoms_enumerated, atoms_enumerated, residues_enumerated


def _write_to_json_to_data_dir(fname: str, dict_to_write: dict):
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    fname = fname.removeprefix('/').removesuffix(FileExt.dot_json.value)
    relpath_json = f'{Path.data_dir.value}/{fname}{FileExt.dot_json.value}'

    with open(relpath_json, 'w') as json_f:
        json.dump(dict_to_write, json_f, indent=4)
    _restore_original_working_dir(cwd)


def read_aa_atoms_yaml() -> Tuple[list, dict]:
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    aas_atoms = dict()
    aas = list()

    with open(Path.aa_atoms_yaml.value, 'r') as stream:
        try:
            atoms_aas = yaml.load(stream, Loader=yaml.Loader)
            aas = atoms_aas[YamlKey.root.value][YamlKey.aas.value]
            aas_atoms = atoms_aas[YamlKey.root.value][YamlKey.atoms_by_aa.value]

        except yaml.YAMLError as exc:
            print(exc)
    _restore_original_working_dir(cwd)
    return aas, aas_atoms


def write_pdb_uniprot_fasta_recs_to_json(recs: dict, filename: str) -> None:
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    fname = filename.removesuffix(FileExt.dot_json.value)
    with open(f'{Path.data_fasta_dir.value}/{fname}{FileExt.dot_json.value}', 'w') as json_f:
        json.dump(recs, json_f, indent=4)  # Works ok (despite warning that it expected SupportsWrite[str], not TextIO).
    _restore_original_working_dir(cwd)


def _remove_null_entries(pdbids_fasta_json: dict):
    pdbids_fasta_json = {k: v for k, v in pdbids_fasta_json.items() if v is not None}
    return pdbids_fasta_json


def read_nonnull_fastas_from_json_to_dict(fname: str) -> dict:
    pdbids_fasta_json = _read_json_from_data_dir(fname=f'{Path.fasta_dir.value}/{fname}')
    pdbids_fasta_json = _remove_null_entries(pdbids_fasta_json)
    return pdbids_fasta_json


def make_api_calls_to_fetch_mmcif_and_write_locally(pdb_id: str, cif_dst_dir: str):
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    non_200_count = 0
    cif_dst_dir = cif_dst_dir.removesuffix('/').removeprefix('/')
    pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
    mmcif_file = f'{cif_dst_dir}/{pdb_id}{FileExt.dot_CIF.value}'
    if os.path.exists(mmcif_file):
        print(f'{mmcif_file} already exists. No API call required.')
    else:
        response = api.call_for_cif_with_pdb_id(pdb_id)
        code = response.status_code
        if code != 200:
            non_200_count += 1
            print(f'Response status code for {pdb_id} is {code}, hence could not read the pdb for this id.')
        with open(mmcif_file, 'w') as file:
            file.write(response.text)
    print(f'{non_200_count} non-200 status codes out of {pdb_id} PDB API calls.')
    _restore_original_working_dir(cwd)


def save_torch_tensor(pt: torch.Tensor, dst_dir: str, pdbid_chain: str):
    os.makedirs(dst_dir, exist_ok=True)
    pt_file = f'{dst_dir}/{pdbid_chain}{FileExt.dot_pt.value}'
    torch.save(pt, pt_file)


def write_tokenised_cif_to_flatfile(pdb_id: str, pdfs: List[pd.DataFrame], dst_data_dir=None, flatfiles=None):
    """
    Write dataframe of single protein, and single chain, with columns CIF.S_seq_id, CIF.S_mon_id, CIF.A_id,
    CIF.A_label_atom_id, ColNames.MEAN_CORR_X, ColNames.MEAN_CORR_Y, ColNames.MEAN_CORR_Z to flat file(s) in local
    relative dir 'src/diffusion/diff_data/tokenised/' or to the top-level `data` dir.
    :param pdb_id: PDB id.
    :param pdfs: List of dataframes to write to flat file(s). One dataframe per polypeptide chain. TODO: hacked to one.
    :param dst_data_dir: Relative path to destination dir of flatfile of tokenised cif. (Will be called from either
    `diffSock/test`, `diffSock/src/preprocessing_funcs` or `diffSock/src/diffusion`. The responsibility for determining
    the relative destination path is left to the caller).
    :param flatfiles: List of file formats (e.g. ['ssv', 'csv', 'tsv'], or string of one format, otherwise just
    one ssv file per protein and chain, by default). E.g. Chain 'A' for PDB id '10J6' is written to `10J6_A.ssv`.
    """
    print(f'PDBid={pdb_id}: write tokenised to ssv')
    for pdf in pdfs:
        if flatfiles is None:
            flatfiles = [FileExt.ssv.value]
        elif isinstance(flatfiles, str):
            flatfiles = [flatfiles]
        cwd = ''  # to return to at end of this function.
        if not dst_data_dir:  # i.e. use the top-level general-use `data` dir & define relpath from data_layer
            # Store cwd to return to at end. Change current dir to data layer:
            print(f'You did not pass any destination dir path for writing the tokenised cif flat flatfile to. '
                  f'Therefore it will be written to the top-level data dir (`diffSock/data/tokenised`).')
            cwd = _chdir_to_data_layer()
            dst_data_dir = Path.data_tokenised_dir.value
        else:
            os.makedirs(dst_data_dir, exist_ok=True)

        chain = pdf[CIF.S_asym_id.value].unique()
        chain = chain[0]
        for flatfile in flatfiles:
            sep = ' '
            if flatfile == FileExt.tsv.value:
                sep = '\t'
            elif flatfile == FileExt.csv.value:
                sep = ','
            dst_data_dir = dst_data_dir.removesuffix('/')
            pdb_id = pdb_id.removesuffix(FileExt.dot_CIF.value)
            pdf.to_csv(path_or_buf=f'{dst_data_dir}/{pdb_id}_{chain}.{flatfile}', sep=sep, index=False)

        # # For a more human-readable set of column-names:
        # pdf_easy_read = pdf.rename(columns={CIF.S_seq_id.value: 'SEQ_ID',
        #                                     CIF.S_mon_id.value: 'RESIDUES',
        #                                     CIF.A_id.value: 'ATOM_ID',
        #                                     CIF.A_label_atom_id.value: 'ATOMS',
        #                                     ColNames.MEAN_CORR_X.value: 'X',
        #                                     ColNames.MEAN_CORR_Y.value: 'Y',
        #                                     ColNames.MEAN_CORR_Z.value: 'Z'})
        # pdf_easy_read.to_csv(path_or_buf=f'../data/tokenised/easyRead_{pdb_id}.tsv', sep='\t',
        # index=False, na_rep='null')

        if not dst_data_dir:
            _restore_original_working_dir(cwd)


def read_tokenised_cif_ssv_to_pdf(pdb_id: str, relpath_tokensd_dir: str) -> List[pd.DataFrame]:
    """
    Read pre-tokenised flatfile (i.e. ssv) of cif for given PDB id, from either `src/diffusion/diff_data/tokenised`or
    top-level `data/tokenised`. The reason for having option of data path is simply a workaround to problems when
    reading from top-level data dir on HPC.
    :param pdb_id: PDB id of protein.
    :param relpath_tokensd_dir: Relative path to the ssv holding the tokenised CIF data.
    E.g. `src/diffusion/diff_data/tokenised`, or `data/tokenised`.
    :return: Pre-tokenised CIF, stored as a ssv flatfile, read back into dataframe.
    """
    os.makedirs(relpath_tokensd_dir, exist_ok=True)
    relpath_tokensd_dir = relpath_tokensd_dir.removesuffix('/').removeprefix('/')
    pattern = fr'{pdb_id}_[A-Z]\{FileExt.dot_ssv.value}'
    ssvs = []
    for f in os.listdir(relpath_tokensd_dir):
        if os.path.isfile(os.path.join(relpath_tokensd_dir, f)) and re.match(pattern, f):
            ssvs.append(f)
    pdfs = []
    for ssv in ssvs:
        path_cif_ssv = f'{relpath_tokensd_dir}/{ssv}'
        print(f'Reading flatfile of tokenised cif: {path_cif_ssv} into dataframe')
        pdf = pd.read_csv(path_cif_ssv, sep=' ')
        nan_indices = pdf[pdf[ColNames.AA_ATOM_LABEL_NUM.value].isna()].index
        if not nan_indices.empty:
            print(f'Row indices with NaN values: {list(nan_indices)}')
        pdfs.append(pdf)
    return pdfs


def read_tokenised_cif_chain_ssv_to_pdf(pdbid_chain: str, relpath_tokensd_dir: str) -> pd.DataFrame:
    """
    Read pre-tokenised flatfile (i.e. ssv) of cif for given PDBid_chain.
    :param pdbid_chain: PDBid, underscore, chain of protein, e.g. '1ECA_A'
    :param relpath_tokensd_dir: Relative path to the ssv holding the tokenised CIF data.
    E.g. `src/diffusion/diff_data/tokenised`, or `data/tokenised`.
    :return: Pre-tokenised CIF for specified chain, stored as a ssv flatfile, read into dataframe.
    """
    relpath_tokensd_dir = relpath_tokensd_dir.removesuffix('/').removeprefix('/')
    path_cif_ssv = f'{relpath_tokensd_dir}/{pdbid_chain}{FileExt.dot_ssv.value}'
    print(f'Reading flatfile of tokenised cif: {path_cif_ssv} into dataframe')
    pdf = pd.read_csv(path_cif_ssv, sep=' ')
    nan_indices = pdf[pdf[ColNames.AA_ATOM_LABEL_NUM.value].isna()].index
    if not nan_indices.empty:
        print(f'Row indices with NaN values: {list(nan_indices)}')
    return pdf



def _read_json_from_data_dir(fname: str) -> dict:
    """
    Read given json file from diffSock/data/{fname} to a Python dict.
    :param fname: File name of json file to read. IMPORTANT: Subdir paths are expected to be included.
    e.g. 'enumeration/fname.json' without starting forward slash.
    :return: The read-in json file, as a Python dict.
    """
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    fname = fname.removeprefix('/').removesuffix(FileExt.dot_json.value)
    relpath_json = f'{Path.data_dir.value}/{fname}{FileExt.dot_json.value}'
    assert os.path.exists(relpath_json)
    try:
        with open(relpath_json, 'r') as json_f:
            my_dict = json.load(json_f)
    except FileNotFoundError:
        print(f'{my_dict} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")

    _restore_original_working_dir(cwd)
    return my_dict


def read_lst_file_from_data_dir(fname):
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    fname = fname.removeprefix('/').removesuffix(FileExt.dot_lst.value)
    relpath_lst = f'{Path.data_dir.value}/{fname}{FileExt.dot_lst.value}'
    assert os.path.exists(relpath_lst)
    try:
        with open(relpath_lst, 'r') as lst_f:
            f = [line.strip() for line in lst_f]
    except FileNotFoundError:
        print(f'{lst_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")

    _restore_original_working_dir(cwd)
    return f


def _manually_write_aa_atoms_to_data_dir(path: str) -> None:
    """
    This function only needs to be run once.
    :param path:
    :return:
    """
    cwd = _chdir_to_data_layer()  # Store cwd to return to at end. Change current dir to data layer
    aa_atoms = {
        'A': ['N', 'CA', 'C', 'O', 'CB'],
        'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
        'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
        'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
        'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'G': ['N', 'CA', 'C', 'O'],
        'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
        'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
        'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
        'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
        'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
        'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
        'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
        'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
        'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
        'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
        'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
        'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
    }
    with open(path, 'w') as json_f:
        json.dump(aa_atoms, json_f, indent=4)
    _restore_original_working_dir(cwd)


# if __name__ == '__main__':

    # copy_cifs_from_bigfilefolder_to_diff_data()
    # clear_diffdatacif_dir()
#     # THIS NEED ONLY BE RUN ONCE, TO GENERATE THE JSON FILE.
#     _manually_write_aa_atoms_to_data_dir(path=Path.data_per_aa_atoms_json.value)

