# Placeholder file for script that will contain all the functions that read/write from/to `data` subdirs, to just make
# the other functions a bit tidier.
import glob
import os


def get_list_of_pdbids_of_locally_downloaded_cifs() -> list:
    cifs = glob.glob(os.path.join('../data/cifs', '*.cif'))
    path_cifs = [cif for cif in cifs if os.path.isfile(cif)]

    pdb_ids = []

    for path_cif in path_cifs:
        cif_basename = os.path.basename(path_cif)
        pdbid = os.path.splitext(cif_basename)[0]
        pdb_ids.append(pdbid)

    return pdb_ids


# def get_list_of_uniprotids_of_locally_downloaded_cifs():


def write_list_to_space_separated_txt_file(list_to_write: list, file_name: str) -> None:
    space_sep_str = ' '.join(list_to_write)
    with open(f'../data/{file_name}', 'w') as f:
        f.write(space_sep_str)

