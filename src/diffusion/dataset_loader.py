#!~/miniconda3/bin/python

import os
import glob
from math import sqrt
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch


def _torch_load_embed_pt(pt_fname: str):
    """
    Load torch embedding `.pt` file.
    :param pt_fname: Filename of `.pt` file to be loaded.
    :return: A Tensor or a dict.
    """
    abs_path = os.path.dirname(os.path.abspath(__file__))
    pt_fname = pt_fname.removesuffix('.pt')
    path_pt_fname = f'../diffusion/diff_data/emb/{pt_fname}.pt'
    abspath_pt_fname = os.path.normpath(os.path.join(abs_path, path_pt_fname))
    assert os.path.exists(abspath_pt_fname), f"'{pt_fname}.pt' does not exist at '{abspath_pt_fname}'"
    pt = None
    try:
        pt = torch.load(abspath_pt_fname, weights_only=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: '{abspath_pt_fname}'.") from e
    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(f"Failed to unpickle file: '{abspath_pt_fname}'. "
                         f"May be corrupted or not valid PyTorch file.") from e
    except EOFError as e:
        raise EOFError(f"Unexpected end of file while loading file: '{abspath_pt_fname}'. "
                       f"It might be incomplete or corrupted.") from e
    except RuntimeError as e:
        raise RuntimeError(f"PyTorch encountered an issue while loading file: '{abspath_pt_fname}'. "
                           f"Details: {str(e)}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading file: '{abspath_pt_fname}'. "
                        f"Error: {str(e)}") from e

    assert pt is not None, f"'{pt_fname}' seems to have loaded successfully but is null..."

    if isinstance(pt, dict):
        assert bool(pt), f"The dict of '{pt_fname}' was read in successfully but seems to be empty..."
    else:
        if isinstance(pt, torch.Tensor):
            assert pt.numel() > 0, f"The tensor of '{pt_fname}' was read in successfully but seems to be empty..."
    return pt


def _chdir_to_dataset_loader() -> str:
    """
    Using relative paths and the possibility of calling this script from different locations necessitates
    temporarily changing current working directory before changing back at the end.
    :return: Current working directory before changing to this one with absolute path.
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f'Path changed from {cwd} to = {os.getcwd()}. (This is intended to be temporary).')
    return cwd


def load_dataset() -> Tuple[List, List]:

    train_list, validation_list = [], []
    abnormal_structures = []  # list any structures with invalid inter-atomic distances of consecutive residues

    print('Starting `load_dataset()`...')
    cwd = _chdir_to_dataset_loader()  # Change current dir. (Store cwd to return to at end).
    tnum = 0
    sum_d2 = 0
    sum_d = 0
    nn = 0

    # GET SSVS IN TOKENISED DIR:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    abspath_tokenised = os.path.normpath(os.path.join(abs_path, 'diff_data/tokenised'))
    path_ssvs = glob.glob(os.path.join(abspath_tokenised, f'*.ssv'))

    for path_ssv in path_ssvs:  # Expecting only one PDBid per line.
        sp = []
        target_pdbid = os.path.basename(path_ssv)
        print(f'Reading in {target_pdbid} ...')
        pdf_target = pd.read_csv(path_ssv, sep=' ')

        # GET COORDINATES TO 2D ARRAY OF (NUM_OF_ATOMS, 3):
        # coords = pdf_target[['mean_corrected_x', 'mean_corrected_y', 'mean_corrected_z']].values
        coords = pdf_target[['A_Cartn_x', 'A_Cartn_y', 'A_Cartn_z']].values

        # GET `atomcodes` VIA 'atom_label_num' COLUMN, WHICH HOLDS ENUMERATED ATOMS VALUES:
        atomcodes = pdf_target['atom_label_num'].tolist()

        # THE FOLLOWING LINE INCREMENTS BY 1 FOR EVERY SUBSEQUENT RESIDUE, SUCH THAT FOR A PDB OF 100 RESIDUES,
        # REGARDLESS OF WHAT RESIDUE THE STRUCTURAL DATA STARTS FROM AND REGARDLESS OF ANY GAPS IN THE SEQUENCE,
        # `aaindices` STARTS FROM 0, INCREMENTS BY 1 AND ENDS AT 99:
        pdf_target['aaindices'] = pd.factorize(pdf_target['S_seq_id'])[0]
        aaindices = pdf_target['aaindices'].tolist()

        # ASSIGN DATAFRAME INDEX OF BACKBONE ATOM POSITION PER RESIDUE IN NEW COLUMN `BBINDICES` FOR REF TO ATOMS:
        atom_positions_rowindex_map = {value: index for index, value in pdf_target['A_id'].items()}
        pdf_target['bbindices'] = pdf_target['bb_atom_pos'].map(atom_positions_rowindex_map)

        # DE-DUPLICATE ROWS ON RESIDUE POSITION (`S_seq_id`) TO GET CORRECT DIMENSION OF `aacodes` and `bbindices`:
        pdf_target_deduped = (pdf_target
                              .drop_duplicates(subset='S_seq_id', keep='first')
                              .reset_index(drop=True))

        # GET `aacodes`, VIA 'aa_label_num' COLUMN, WHICH HOLDS ENUMERATED RESIDUES VALUES:
        aacodes = pdf_target_deduped['aa_label_num'].tolist()

        bbindices = pdf_target_deduped['bbindices'].tolist()

        # ONLY INCLUDE PROTEINS WITHIN A CERTAIN SIZE RANGE:
        if len(aacodes) < 10 or len(aacodes) > 500:
            print(f'{target_pdbid} mmCIF is {len(aacodes)} residues long. It is not within the chosen range 10-500 '
                  f'residues, so will be excluded.')
            continue

        # READ PRE-COMPUTED EMBEDDING OF THIS PROTEIN:
        pdb_embed = _torch_load_embed_pt(pt_fname=target_pdbid)

        # AND MAKE SURE IT HAS SAME NUMBER OF RESIDUES AS THE PARSED-TOKENISED SEQUENCE FROM MMCIF:
        assert pdb_embed.size(1) == len(aacodes), 'Size of embedding does not match length of aacodes.'

        # ONE BACKBONE ATOM (ALPHA-CARBON) PER RESIDUE. SO `len(bbindices)` SHOULD EQUAL NUMBER OF RESIDUES:
        assert len(aacodes) == len(bbindices), 'Length of aacodes does not match length of bbindices.'

        # MAKE SURE YOU HAVE AT LEAST THE MINIMUM NUMBER OF EXPECTED ATOMS IN MMCIF DATA:
        min_num_atoms_expected_per_residue = 4  # GLYCINE HAS 4 NON-H ATOMS: 1xO, 2xC, 1xN, 5xH.
        min_num_expected_atoms = len(bbindices) * min_num_atoms_expected_per_residue
        # THIS IS THE NUMBER OF ATOMS (AS ONE ROW PER ATOM IN 'aaindices' DUE TO OUTER-JOIN):
        num_of_atoms_in_cif = len(aaindices)

        # ASSUME PROTEIN WILL NEVER BE 100 % GLYCINES (OTHERWISE I'D USE `<=` INSTEAD OF `<`):
        if num_of_atoms_in_cif < min_num_expected_atoms:
            print("WARNING: Too many missing atoms in ", target_pdbid, len(aacodes), len(aaindices))
            continue

        aacodes = np.asarray(aacodes, dtype=np.uint8)
        atomcodes = np.asarray(atomcodes, dtype=np.uint8)
        bbindices = np.asarray(bbindices, dtype=np.int16)
        aaindices = np.asarray(aaindices, dtype=np.int16)
        target_coords = np.asarray(coords, dtype=np.float32)
        target_coords -= target_coords.mean(0)

        assert len(aacodes) == target_coords[bbindices].shape[0]

        # SANITY-CHECKING INTER-ATOMIC DISTANCES FOR CONSECUTIVE RESIDUES:
        diff = target_coords[1:] - target_coords[:-1]
        distances = np.linalg.norm(diff, axis=1)
        # assert distances.min() >= 0.5, f'Error: Overlapping atoms detected in {target_pdbid}'
        # assert distances.max() <= 3.5, f'Error: Abnormally large gaps in {target_pdbid}'

        print(target_coords.shape, target_pdbid, len(aacodes), distances.min(), distances.max())

        if distances.min() < 1.0 or distances.max() > 2.0:
            abnormal_structures.append((target_pdbid, distances.min(), distances.max()))

        sp.append((aacodes, atomcodes, aaindices, bbindices, target_pdbid, target_coords))

        # Choose every 10th sample for validation
        if tnum % 10 == 0:
            validation_list.append(sp)
        else:
            train_list.append(sp)
        tnum += 1

        # CALCULATE THE STANDARD DEVIATION:
        sum_d2 += (target_coords ** 2).sum()
        sum_d += np.sqrt((target_coords ** 2).sum(axis=-1)).sum()
        nn += target_coords.shape[0]
        sigma_data = sqrt((sum_d2 / nn) - (sum_d / nn) ** 2)
        print(f'Data s.d. = {sigma_data:.2f}')
        print(f'Data unit var scaling = {(1 / sigma_data):.2f}')

    if abnormal_structures:
        print('Abnormal structures detected:')
        for pdbid, min_dist, max_dist in abnormal_structures:
            print(f'{pdbid}: Min {min_dist:.2f}, Max {max_dist:.2f}')

    os.chdir(cwd)  # restore original working directory
    print('Finished `load_dataset()`...')
    return train_list, validation_list


if __name__ == '__main__':

    from time import time
    start_time = time()

    _train_list, _validation_list = load_dataset()

    time_taken = time() - start_time

    from pathlib import Path
    _abs_path = os.path.dirname(os.path.abspath(__file__))
    abspath_emb = os.path.normpath(os.path.join(_abs_path, 'diff_data/tokenised'))
    path = Path(abspath_emb)
    ssv_count = sum(1 for file in path.rglob("*.ssv"))
    print(f'Datasets using {ssv_count} PDBs loaded in {time_taken:.2f} seconds.')
