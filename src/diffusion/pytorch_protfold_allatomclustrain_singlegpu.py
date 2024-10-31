#!/home/jones/miniconda3/bin/python
"""
DJ's diffusion method for training and inference of RNA structures from primary sequence, adapted here for proteins.

General notes:
- `pdf_` refers to pandas dataframe.

"""

import sys
import os
from enum import Enum
import time
import random
from math import sqrt, log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from nndef_protfold_atompyt2 import DiffusionNet

from data_layer import data_handler as dh
from src.preprocessing_funcs import tokeniser as tk
from src.preprocessing_funcs.tokeniser import ColNames
from src.preprocessing_funcs.cif_parser import CIF

bb_atoms = ["C", "CA", "N", "O", "OXT"]
bb_atoms_two = ["C", "N"]  # the peptide bond carbonyl and amino nitrogen
sc_atoms = ["CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2",
            "CH2", "CZ", "CZ2", "CZ3", "ND1", "ND2", "NE", "NE1", "NE2", "NH1",
            "NH2", "NZ", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OG2", "OH",
            "SD", "SG"]


class Cols(Enum):
    AA_LABEL_NUM = 'aa_label_num'       # Enumerated residues, mapped from `A_label_comp_id`.
    ATOM_LABEL_NUM = 'atom_label_num'   # Enumerated atoms, mapped from `A_label_atom_id`.
    BB_INDEX = 'bb_index'               # The position of one of the backbone atoms. C-alpha ('CA') is chosen here.
    MEAN_COORDS = 'mean_xyz'            # Mean of x y z coordinates for each atom.
    MEAN_CORR_X = 'mean_corrected_x'    # x coordinates for each atom subtracted by the mean of xyz coordinates.
    MEAN_CORR_Y = 'mean_corrected_y'    # (as above) but for y coordinates.
    MEAN_CORR_Z = 'mean_corrected_z'    # (as above) but for z coordinates.


# INPUT FILE NAMES:
# json file names:
AA_ATOMS_CODES = 'aas_atoms_enumerated'
AA_ATOMS_CODES_NO_H = 'aas_atoms_enumerated_no_hydrogens'
ATOMS_ONLY_CODES = 'unique_atoms_only_enumerated'
ATOMS_ONLY_CODES_NO_H = 'unique_atoms_only_enumerated_no_hydrogens'
AAS_CODES_1_LETTER = 'FASTA_aas_enumerated'
AAS_CODES_3_LETTER = 'aas_enumerated'

# lst file name:
PROT_TRAIN_CLUSTERS = 'prot_train_clusters'

# paths:
PATH_TO_CIF_DIR = '../src/diffusion/data/cif/'
PATH_TO_TOKENISED_DIR = 'data/tokenised/'
PATH_TO_EMB_DIR = 'data/emb/'
# OUTPUT FILE NAMES:
PROT_E2E_MODEL_PT = 'prot_e2e_model.pt'
# CAN BE INPUT OR OUTPUT:
PROT_E2E_MODEL_TRAIN_PT = 'prot_e2e_model_train.pt'
CHECKPOINT_PT = 'checkpoint.pt'


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# BATCH_SIZE = 32
BATCH_SIZE = 8
# NSAMPLES = 24
NSAMPLES = 12
SIGDATA = 16
        
RESTART_FLAG = True
FINETUNE_FLAG = False


def _read_lst_file_from_src_diff_dir(fname):
    fname = fname.removeprefix('/').removesuffix('.lst')
    relpath_lst = f'{fname}.lst'
    assert os.path.exists(relpath_lst)
    try:
        with open(relpath_lst, 'r') as lst_f:
            f = [line.strip() for line in lst_f]
    except FileNotFoundError:
        print(f'{lst_f} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return f


def _impute_missing_coords(pdf_to_impute):
    """
    Impute missing values of the mean x, y, z structure coordinates with 0s.
    :param pdf_to_impute: Dataframe to impute missing data.
    :return: Imputed dataframe.
    """
    pdf_to_impute[[ColNames.MEAN_CORR_X.value,
                   ColNames.MEAN_CORR_Y.value,
                   ColNames.MEAN_CORR_Z.value]] = (pdf_to_impute[[ColNames.MEAN_CORR_X.value,
                                                                  ColNames.MEAN_CORR_Y.value,
                                                                  ColNames.MEAN_CORR_Z.value]].fillna(0, inplace=False))
    return pdf_to_impute


# Load dataset function remains the same
def load_dataset():

    train_list = []
    validation_list = []
    tnum = 0

    # atokendict = {"OP3": 0, "P": 1, "OP1": 2, "OP2": 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9,
    #               "C2'": 10, "O2'": 11, "C1'": 12, "N9": 13, "C8": 14, "N7": 15, "C5": 16, "C6": 17, "O6": 18,
    #               "N1": 19, "C2": 20, "N2": 21, "N3": 22, "C4": 23, "O2": 24, "N4": 25, "N6": 26, "O4": 27}
    atokendict = dh.read_atom_enumeration_mapping(fname=ATOMS_ONLY_CODES_NO_H)
    # # Or if using amino-acid-atom pairings:
    # atokendict = dh.read_atom_enumeration_mapping(fname=AA_ATOMS_CODES_NO_H)

    # ntnumdict = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    aanumdict = dh.read_fasta_aa_enumeration_mapping(fname=AAS_CODES_3_LETTER)

    sum_d2 = 0
    sum_d = 0
    nn = 0
    targetfile = _read_lst_file_from_src_diff_dir(fname=PROT_TRAIN_CLUSTERS)
    for line in targetfile:
        targets = line.rstrip().split()

        # if you know all these cif files are already in the local data dir, then no need to run following line:
        # The cwd is 'src/diffusion'. Writes to 'src/diffusion/data/cif/' dir.
        dh.make_api_calls_to_fetch_mmcif_and_write_locally(pdb_ids=targets, dst_path=PATH_TO_CIF_DIR)

        sp = []
        for target in targets:
            cif_tokenised_ssv = f'{PATH_TO_TOKENISED_DIR}{target}.ssv'
            if os.path.exists(cif_tokenised_ssv):
                pdf_target = dh.read_tokenised_cif_ssv_to_pdf(pdb_id=target, use_subdir=True)
            else:
                pdf_target = tk.parse_tokenise_cif_write_flatfile(pdb_ids=target, dst_path_for_tokenised='data/tokenised')
            print(pdf_target.columns.tolist())  # Keeping track of what columns the df has here.
            pdf_target = _impute_missing_coords(pdf_target)

            # TODO: Add new column that indicates whether the atom is main-chain or side-chain, primarily to use for
            # getting confirmation from DJ/SK that they are correct. Would potentially be useful for making an explicit
            # filtration or to create two different data subsets, one with only backbones and one with only side-chains
            # or backbone only vs all, etc.

            coords = pdf_target[[ColNames.MEAN_CORR_X.value,
                                 ColNames.MEAN_CORR_Y.value,
                                 ColNames.MEAN_CORR_Z.value]].values  # should be list of 3-element numpy arrays
            aaindex = -1  # replacing `ntindex`

            pdf_target_deduped = pdf_target.drop_duplicates(subset=CIF.S_seq_id.value, keep='first').reset_index(drop=True)

            aacodes = [aanumdict[aa] for aa in pdf_target_deduped[CIF.S_mon_id.value].tolist()]  # the enumeration of the amino acid (i.e. 0-19)
            aaindices = pdf_target_deduped[CIF.S_seq_id.value].tolist()  # the index of each amino acid (should increase and repeat for several rows, as each aa has several atoms)

            atomindex = 0
            lastnid = None

            length = aaindex + 1

            if length < 10 or length > 500:
                continue
            pdb_embed = torch.load(f'{PATH_TO_EMB_DIR}{target}.pt')
            assert pdb_embed.size(1) == length  # This is the length of protein (i.e. number of residues)

            # In the RNA diffusion script, the minimum allowed number of atoms per base is 6.
            # The smallest base, uracil has 12 atoms (cg4), but 40 atoms for a complete uracil nucleotide including the
            # nitrogenous base, ribose sugar, and phosphate group in RNA. So, does this mean DJ is saying it's ok to
            # include those RNA cifs with up to 34 of 40 atoms of a base missing?
            assert length == len(bbindices)
            min_num_atoms_expected_per_aa = 5  # i.e. 5 non-H atoms in smallest residue glycine: 2xO, 2xC, 1xN, 5xH.
            # bbindices should be one backbone atom per residue, so len(bbindices) should be number of residues.
            # Therefore min number of expected atoms is len(bbindices) * min number of non-H atoms per residue.
            min_num_expected_atoms = len(bbindices) * min_num_atoms_expected_per_aa
            num_of_atoms_in_cif = len(aaindices)  # This is number of atoms, due to outer-join, mimicing DJ's RNA code.

            # Assuming the protein will never be 100% Glycines (otherwise I would use <= instead of <).
            if num_of_atoms_in_cif < min_num_expected_atoms:
                print("WARNING: Too many missing atoms in ", target, length, len(aaindices))
                continue

            aacodes = np.asarray(aacodes, dtype=np.uint8)
            atomcodes = np.asarray(atomcodes, dtype=np.uint8)
            bbindices = np.asarray(bbindices, dtype=np.int16)
            aaindices = np.asarray(aaindices, dtype=np.int16)

            target_coords = np.asarray(coords, dtype=np.float32)
            target_coords -= target_coords.mean(0)

            assert length == target_coords[bbindices].shape[0]

            sum_d2 += (target_coords ** 2).sum()
            sum_d += np.sqrt((target_coords ** 2).sum(axis=-1)).sum()
            nn += target_coords.shape[0]

            diff = target_coords[1:] - target_coords[:-1]
            distances = np.linalg.norm(diff, axis=1)

            print(target_coords.shape, target, length, distances.min(), distances.max())

            sp.append((aacodes, atomcodes, aaindices, bbindices, target, target_coords))

        # Choose every 10th sample for validation
        if tnum % 10 == 0:
            validation_list.append(sp)
        else:
            train_list.append(sp)
        tnum += 1
        
    sigma_data = sqrt((sum_d2 / nn) - (sum_d / nn) ** 2)
    print(f'Data s.d. = , {sigma_data}')
    print(f'Data unit var scaling = , {1 / sigma_data}')

    return train_list, validation_list


# Superpose coordinates (superposes c1 on c2)
def lsq_fit(c1, c2):
    """
    To compare two sets of 3D coordinates in a way that is rotation-invariant, this function performs the rotation
    that minimises any rotational differences, so that subsequent comparisons are
    A rigid transformation fitting, using least squares method for minimising the distance between two sets of
    coordinates. Minimisation is achieved using SVD. (This involves aligning points, via rotation, rather than
    fitting a model to data as in the application of least squares that I am more used to encountering). Note the
    sum of squared errors does not need to be explicitly calculated because the use of SVD shortcuts to the
    optimal rotation matrix, so its performed implicitly.
    :param c1: Set of 3D points.
    :param c2: Set of 3D points.
    :return: The set of points from `c1` after they've been rotated to align as closely as possible with `c2`.
    Best-fit transformation of `c1` coordinates onto `c2` coordinates, where only a rotation (no scaling or
    reflection) has been applied.
    """
    with torch.no_grad():
        P = c1.transpose(1, 2)
        Q = c2.transpose(1, 2)
        P_mean = P.mean(dim=2, keepdim=True)
        Q_mean = Q.mean(dim=2, keepdim=True)
        P = P - P_mean
        Q = Q - Q_mean

        if P.size(0) == 1 and Q.size(0) > 1:
            P = P.expand(Q.size(0), -1, -1)

        cov = torch.matmul(P, Q.transpose(1, 2))

        # Find optimal rotation matrix using SVD (which minimises squared distances between corresponding points):
        try:
            U, S, Vh = torch.linalg.svd(cov)
        except RuntimeError:
            return None

        #  Applying the rotation to align c1 with c2:
        V = Vh.transpose(-2, -1)
        d = torch.eye(3, device=P.device).repeat(P.size(0),1,1)
        d[:, 2, 2] = torch.det(torch.matmul(V, U.transpose(-2, -1)))

        rot = torch.matmul(torch.matmul(V, d), U.transpose(-2, -1))
        rot_P = torch.matmul(rot, P)

    return (rot_P + Q_mean).transpose(1, 2)


def random_rotation_matrices(N):
    """Generates N random rotation matrices."""
    axes = np.random.randn(N, 3)
    axes /= np.linalg.norm(axes, axis=1)[:, np.newaxis]  # normalise each vector
    angles = np.random.uniform(0, 2 * np.pi, N)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    one_minus_cos = 1 - cos_a

    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = cos_a + x**2 * one_minus_cos
    R[:, 0, 1] = x * y * one_minus_cos - z * sin_a
    R[:, 0, 2] = x * z * one_minus_cos + y * sin_a
    R[:, 1, 0] = y * x * one_minus_cos + z * sin_a
    R[:, 1, 1] = cos_a + y**2 * one_minus_cos
    R[:, 1, 2] = y * z * one_minus_cos - x * sin_a
    R[:, 2, 0] = z * x * one_minus_cos - y * sin_a
    R[:, 2, 1] = z * y * one_minus_cos + x * sin_a
    R[:, 2, 2] = cos_a + z**2 * one_minus_cos
    return R


class DMPDataset(Dataset):

    def __init__(self, sample_list, augment=True):
        self.sample_list = sample_list
        self.augment = augment
        # self.mse_sum = 0
        # self.mse_count = 0
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, tn):
        if self.augment:
            sample = random.choice(self.sample_list[tn])
        else:
            sample = self.sample_list[tn][0]
        # ntseq = sample[0]
        aaseq = sample[0]
        atomcodes = sample[1]
        # ntindices = sample[2]
        aaindices = sample[2]
        bbindices = sample[3]
        target = sample[4]
        target_coords = sample[5]

        embed = torch.load(f'{PATH_TO_EMB_DIR}{target}.pt')
        
        # length = ntseq.shape[0]
        length = aaseq.shape[0]

        if FINETUNE_FLAG:
            croplen = 20
        else:
            croplen = random.randint(10, min(20, length))

        if self.augment and length > croplen:
            lcut = random.randint(0, length-croplen)
            # ntseq = ntseq[lcut:lcut+croplen]
            aaseq = aaseq[lcut:lcut + croplen]
            bbindices = bbindices[lcut: lcut + croplen]
            bb_coords = target_coords[bbindices]
            embed = embed[:, lcut: lcut + croplen]
            # mask = np.logical_and(ntindices >= lcut, ntindices < lcut+croplen)
            mask = np.logical_and(aaindices >= lcut, aaindices < lcut + croplen)
            atomcodes = atomcodes[mask]
            # ntindices = ntindices[mask] - lcut
            aaindices = aaindices[mask] - lcut
            target_coords = target_coords[mask]
            length = croplen
        else:
            bb_coords = target_coords[bbindices]

        if target_coords.shape[0] < 10:
            # print(target, length, ntindices)
            print(target, length, aaindices)

        noised_coords = target_coords - target_coords.mean(axis=0)

        # Original coordinates and replicating for N sets (N, L, 3)
        batched_coords = np.repeat(noised_coords[np.newaxis, :, :], NSAMPLES, axis=0)

        if self.augment:
            # Generate N rotation matrices (N, 3, 3)
            rotation_matrices = random_rotation_matrices(NSAMPLES)
            translations = np.random.randn(NSAMPLES,1,3)
            # Apply rotations using einsum for batch matrix multiplication
            batched_coords = np.einsum('nij,nkj->nki', rotation_matrices, batched_coords) + translations
            #  distribution = torch.distributions.Beta(1, 8)
            distribution = torch.distributions.Uniform(0, 1)
            tsteps = distribution.sample((NSAMPLES,))
        else:
            tsteps = torch.arange(0, 1, 1 / NSAMPLES)

        sig_max_r7 = (SIGDATA * 10) ** (1/7)
        sig_min_r7 = 4e-4 ** (1/7)
        noise_levels = (sig_max_r7 + tsteps * (sig_min_r7 - sig_max_r7)) ** 7

        # ntcodes = torch.from_numpy(ntseq.copy()).long()
        aacodes = torch.from_numpy(aaseq.copy()).long()
        bb_coords = torch.from_numpy(bb_coords).float()
        # ntindices = torch.from_numpy(ntindices.copy()).long()
        aaindices = torch.from_numpy(aaindices.copy()).long()
        atomcodes = torch.from_numpy(atomcodes.copy()).long()
        target_coords = torch.from_numpy(target_coords.copy()).unsqueeze(0)

        batched_coords = torch.from_numpy(batched_coords).float()
        
        noise = torch.randn_like(batched_coords)

        # print(noise_levels.size(), noise.size(), batched_coords.size())
        noised_coords = noise_levels.view(NSAMPLES, 1, 1) * noise + batched_coords

        # sample = (embed, noised_coords, noise_levels, noise, ntcodes, atomcodes, ntindices, bb_coords, target_coords, target)
        sample = (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords, target)

        return sample


def main():
    global BATCH_SIZE
    
    # Create neural network model
    network = DiffusionNet(seqwidth=1024,
                           atomwidth=128,
                           seqheads=16,
                           atomheads=8,
                           seqdepth=6,
                           atomdepth=3,
                           cycles=2).cuda()

    # Load the dataset
    print("Loading data...")
    train_list, validation_list = load_dataset()

    ntrain = len(train_list)
    nvalidation = len(validation_list)
    
    dmp_train_data = DMPDataset(train_list, augment=True)
    dmp_val_data = DMPDataset(validation_list, augment=False)

    # Trivial collate function
    def my_collate(batch):
        return batch

    train_data_loader = DataLoader(dataset=dmp_train_data, 
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=4,
                                   pin_memory=True,
                                   collate_fn=my_collate)

    val_data_loader = DataLoader(dataset=dmp_val_data, 
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 collate_fn=my_collate)

    if FINETUNE_FLAG:
        max_lr = 1e-5
    else:
        max_lr = 3e-4

    val_err_min = 1e32
    start_epoch = 0
    max_epochs = 2000

    if RESTART_FLAG:
        try:
            pretrained_dict = torch.load(PROT_E2E_MODEL_TRAIN_PT, map_location='cuda')
            model_dict = network.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            network.load_state_dict(pretrained_dict, strict=False)
        except:
            pass

        try:
            checkpoint = torch.load(CHECKPOINT_PT)
            start_iteration = checkpoint['iteration']
            val_err_min = checkpoint['val_err_min']
            print("Checkpoint file loaded.")
        except:
            pass

    optimizer = torch.optim.RAdam(network.parameters(), lr=max_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=10)
    # Initialize the GradScaler to allow float16 processing
    scaler = GradScaler()
        
    print("Starting training...", flush=True)

    # Process one sample and return loss
    def calculate_sample_loss(sample):
        """
        Calculate loss for a single sample, combining 3 difference loss components in a weight ed sum of backbone loss,
        confidence loss and difference loss.
        :param sample: Tuple of the following 10 elements: `embed`, `noised_coords`, `noise_levels`, `noise`,
        `aacodes`, `atomcodes`, `aaindices`, `bb_coords`, `target_coords` and `target`. All of which are used except
        for `noise` (sample[3]) and `target` (sample[9]).
        :return: the combined loss.
        """
        # I'M NOT 100% CLEAR ON THIS NON_BLOCKING ARGUMNET. DOES IT REDUCE COMPUTATION TIME, IN ASYNCHRONOUS CONTEXT ?
        # (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords, target)
        inputs = sample[0].cuda(non_blocking=True)  # `embed` (pLM embedding model?)
        noised_coords = sample[1].cuda(non_blocking=True)  # prediction after noise added ?
        noise_levels = sample[2].cuda(non_blocking=True)  # how much noise ?
        # ntcodes = sample[4].cuda(non_blocking=True)
        aacodes = sample[4].cuda(non_blocking=True)  # Enumeration of amino acids (0-19)
        atomcodes = sample[5].cuda(non_blocking=True)  # Enumeration of atoms (0-37 or 0-186)
        # ntindices = sample[6].cuda(non_blocking=True)
        aaindices = sample[6].cuda(non_blocking=True)  # Position of amino acid in protein.
        bb_coords = sample[7].cuda(non_blocking=True)  # X,Y,Z coordinates of the chosen backbone atoms (`CA`).
        target_coords = sample[8].cuda(non_blocking=True)  # X,Y,Z coordinates of all of the other atoms ? Or
        # specifically non-backbone atoms.. Is it possible to know exactly which atoms are definitely from side-chains?

        # pred_denoised, pred_coords, pred_confs = network(inputs, ntcodes, atomcodes, ntindices, noised_coords, noise_levels)
        pred_denoised, pred_coords, pred_confs = network(inputs, aacodes, atomcodes, aaindices, noised_coords, noise_levels)

        predmap = torch.cdist(pred_coords, pred_coords)  # What does this do. What is this for ?
        bb_coords = bb_coords.unsqueeze(0)
        targmap = torch.cdist(bb_coords, bb_coords)

        bb_loss = F.mse_loss(predmap, targmap)

        diffmap = (targmap - predmap).abs().squeeze(0)
        incmask = (targmap < 15.0).float().squeeze(0).fill_diagonal_(0)
        lddt = (0.25 * ((diffmap < 0.5).float() +
                        (diffmap < 1.0).float() +
                        (diffmap < 2.0).float() +
                        (diffmap < 4.0).float()) * incmask).sum(dim=0) / torch.clip(incmask.sum(dim=0), min=1)

        conf_loss = (torch.sigmoid(pred_confs).squeeze(0) - lddt).abs().mean()

        rot_targets = lsq_fit(target_coords, pred_denoised)

        nwts = (noise_levels.pow(2) + SIGDATA * SIGDATA) / (noise_levels + SIGDATA).pow(2)
        diff_loss = ((rot_targets - pred_denoised).pow(2).sum(dim=2).mean(dim=1) * nwts).mean()

        loss = bb_loss + 0.01 * conf_loss + diff_loss

        # Return zero if NaN
        if loss != loss:
            loss = 0

        return loss

    for epoch in range(start_epoch, max_epochs):
        last_time = time.time()
        train_err = 0.0
        train_samples = 0
        network.train()
        for sample_batch in train_data_loader:
            optimizer.zero_grad()

            batch_loss = 0
            for sample in sample_batch:
                batch_loss = batch_loss + calculate_sample_loss(sample)

            # Scale the loss and call backward()
            scaler.scale(batch_loss / len(sample_batch)).backward()
            scaler.step(optimizer)
            scaler.update()

            train_err += batch_loss.item()
            train_samples += len(sample_batch)

        train_err /= train_samples

        # Run validation samples
        network.eval()
        val_err = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_data_loader:
                for sample in batch:
                    val_err += calculate_sample_loss(sample)
                    val_samples += 1

            val_err /= val_samples
            #  scheduler.step(val_err)

            print(f"Epoch {epoch}, train loss: {train_err:.4f}, val loss: {val_err:.4f}")
            print(f"Time taken = {time.time() - last_time:.2f}", flush=True)
            last_time = time.time()

            if val_err < val_err_min:
                val_err_min = val_err
                torch.save(network.state_dict(), PROT_E2E_MODEL_PT)
                print("Saving model...", flush=True)
                    
            torch.save(network.state_dict(), PROT_E2E_MODEL_TRAIN_PT)

            torch.save({
                'epoch': epoch,
                'val_err_min': val_err_min,
            }, CHECKPOINT_PT)


if __name__ == "__main__":
    main()
