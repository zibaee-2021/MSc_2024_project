#!~/miniconda3/bin/python
"""
DJ's diffusion method for training and inference of RNA structures from primary sequence, adapted here for proteins.
General notes:
`pdf_` in any variable name refers to it being Pandas dataframe.
"""

import sys
import os
from typing import List, Tuple
from numpy.typing import NDArray
# from enum import Enum
import time
import random
from math import sqrt, log
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from nndef_protfold_atompyt2 import DiffusionNet

from data_layer import data_handler as dh
from src.preprocessing_funcs import tokeniser as tk
# from src.enums import ColNames, CIF

from src.losses import loss_plotter

# `rp_` stands for relative path:
# class Path(Enum):
#     rp_diffdata_emb_dir = 'diff_data/emb'


# class Filename(Enum):
#     # OUTPUT FILE NAMES:
#     prot_e2e_prot_model_pt = 'prot_e2e_model.pt'
#     # CAN BE INPUT OR OUTPUT:
#     prot_e2e_model_train_pt = 'prot_e2e_model_train.pt'
#     checkpoint_pt = 'checkpoint.pt'


# class FileExt(Enum):
    # dot_pt = '.pt'


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# BATCH_SIZE = 32
BATCH_SIZE = 8
# NSAMPLES = 24
NSAMPLES = 12
SIGDATA = 16

RESTART_FLAG = False  # False to build first train model
FINETUNE_FLAG = False


def _atomic_torch_save(data: dict, pt_fname: str) -> None:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_temp = f"pt_files/{pt_fname.removesuffix('.pt') + '.tmp'}"
    abspath_temp = os.path.normpath(os.path.join(abs_path, path_temp))
    torch.save(data, abspath_temp)
    path_pt = f"pt_files/{pt_fname}"
    abspath_pt = os.path.normpath(os.path.join(abs_path, path_pt))
    os.replace(abspath_temp, abspath_pt)
    print(f"Saved model '{pt_fname}'", flush=True)


def load_dataset(targetfile_lst_path: str) -> Tuple[List, List]:
    train_list, validation_list = tk.load_dataset(targetfile_lst_path)
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
        aacodes = sample[0]
        atomcodes = sample[1]
        # ntindices = sample[2]
        aaindices = sample[2]
        bbindices = sample[3]
        target = sample[4]
        target_coords = sample[5]

        # embed = torch.load(f'{Path.rp_diffdata_emb_dir.value}/{target}{FileExt.dot_pt.value}')
        abs_path = os.path.dirname(os.path.abspath(__file__))
        embed_pt = f'diff_data/emb/{target}.pt'
        abspath_embed_pt = os.path.normpath(os.path.join(abs_path, embed_pt))
        assert os.path.exists(abspath_embed_pt), f'{embed_pt} is missing!'
        embed = torch.load(abspath_embed_pt, weights_only=True)
        linear_layer = nn.Linear(768, 1024, bias=False)
        embed = linear_layer(embed)
        embed = embed.detach()

        # length = ntseq.shape[0]
        prot_length = aacodes.shape[0]

        if FINETUNE_FLAG:  # False
            # croplen = 20
            peptide_window = 20
        else:
            # croplen = random.randint(10, min(20, length))
            peptide_window = random.randint(10, min(20, prot_length))

        # print(f'croplen={croplen}')
        # print(f'peptide_window={peptide_window}')

        if self.augment and prot_length > peptide_window:
            # lcut = random.randint(0, length - croplen)
            start = random.randint(0, prot_length - peptide_window)
            end = start + peptide_window
            aacodes = aacodes[start: end]
            bbindices = bbindices[start: end]
            bb_coords = target_coords[bbindices]  #
            embed = embed[:, start: end]
            mask = np.logical_and(aaindices >= start, aaindices < end)
            atomcodes = atomcodes[mask]
            aaindices = aaindices[mask] - start  # Why subtract from aaindices, but not others?
            target_coords = target_coords[mask]
            prot_length = peptide_window

        else:
            bb_coords = target_coords[bbindices]

        if target_coords.shape[0] < 10:
            # print(target, length, ntindices)
            print(f'target={target}, length={prot_length}, aaindices={aaindices}')

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
        aacodes = torch.from_numpy(aacodes.copy()).long()
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


def main(targetfile_lst_path: str) -> Tuple[NDArray[np.int16], NDArray[np.float16], NDArray[np.float16]]:

    if not RESTART_FLAG:
        print(f"`RESTART_FLAG` is False, hence don't try to use the pre-built models: 'prot_e2e_model_train.pt' and "
              f"'checkpoint.pt'")
    else:
        assert os.path.exists('prot_e2e_model_train.pt'), ("Expected 'prot_e2e_model_train.pt' model file is absent. "
                                                           "Cannot proceed until this is addressed.")
    print(f'FINETUNE_FLAG={FINETUNE_FLAG}')

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
    train_list, validation_list = load_dataset(targetfile_lst_path)

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
                                   # num_workers=0,
                                   pin_memory=True,
                                   collate_fn=my_collate)

    val_data_loader = DataLoader(dataset=dmp_val_data, 
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=4,
                                 # num_workers=0,
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
            # pretrained_dict = torch.load(Filename.prot_e2e_model_train_pt.value, map_location='cuda')
            pt_fname = 'prot_e2e_model_train.pt'

            pretrained_dict = torch.load(pt_fname, map_location='cuda', weights_only=True)
            model_dict = network.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            network.load_state_dict(pretrained_dict, strict=False)
        except:
            print("Could not read in a pretrained 'prot_e2e_model_train.pt' model.)")
            pass

        try:
            # checkpoint = torch.load(Filename.checkpoint_pt.value)
            checkpoint = torch.load('checkpoint.pt', weights_only=True)
            start_epoch = checkpoint['epoch']
            val_err_min = checkpoint['val_err_min']
            print("checkpoint.pt file loaded.")
        except:
            print("Could not read in a pretrained 'checkpoint.pt' model ")
            pass

    optimizer = torch.optim.RAdam(network.parameters(), lr=max_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=10)
    # Initialize the GradScaler to allow float16 processing
    scaler = GradScaler()
        
    print('Starting training...', flush=True)

    # Process one sample and return loss
    def calculate_sample_loss(_sample: List[torch.Tensor]) -> torch.FloatTensor:
        """
        Calculate loss for a single sample, combining 3 difference loss components in a weighted sum of backbone loss,
        confidence loss and difference loss.
        :param _sample: Tuple of the following 10 elements: `embed`, `noised_coords`, `noise_levels`, `noise`,
        `aacodes`, `atomcodes`, `aaindices`, `bb_coords`, `target_coords` and `target`. All of which are used except
        for `noise` (sample[3]) and `target` (sample[9]).
        :return: the combined loss.
        """
        # I'M NOT 100% CLEAR ON THIS NON_BLOCKING ARGUMENT. DOES IT REDUCE COMPUTATION TIME, IN ASYNCHRONOUS CONTEXT ?
        # (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords, target)
        inputs = _sample[0].cuda(non_blocking=True)  # `embed` is pLM embedding tensor
        noised_coords = _sample[1].cuda(non_blocking=True)  # prediction after noise added ?
        noise_levels = _sample[2].cuda(non_blocking=True)  # how much noise to add ??
        # ntcodes = sample[4].cuda(non_blocking=True)
        aacodes = _sample[4].cuda(non_blocking=True)  # Enumeration of amino acids (0-19)
        atomcodes = _sample[5].cuda(non_blocking=True)  # Enumeration of atoms (0-37 or 0-186)
        # ntindices = sample[6].cuda(non_blocking=True)
        aaindices = _sample[6].cuda(non_blocking=True)  # 0-indexed position of amino acid in protein (as seqres).
        bb_coords = _sample[7].cuda(non_blocking=True)  # X,Y,Z coordinates of the chosen backbone atoms (`CA`).
        target_coords = _sample[8].cuda(non_blocking=True)  # X,Y,Z coordinates of all the other atoms ? Or
        # print(f'Protein PDBid={_sample[9]}. inputs.shape={inputs.shape}. len(aacodes)={len(aacodes)}. '
        #       f'len(aaindices)={len(aaindices)}. len(bb_coords)={len(bb_coords)}.')

        # pred_denoised, pred_coords, pred_confs = network(inputs, ntcodes, atomcodes, ntindices, noised_coords, noise_levels)
        pred_denoised, pred_coords, pred_confs = network(inputs, aacodes, atomcodes, aaindices, noised_coords, noise_levels)

        predmap = torch.cdist(pred_coords, pred_coords)  # What does this do? What is this for ?
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

    # SAVING LOSSES TO ARRAY, TO WRITE TO FILE AND PLOT:
    epochs = np.zeros(max_epochs, dtype=np.int16)
    train_losses = np.zeros(max_epochs, dtype=np.float16)
    val_losses = np.zeros(max_epochs, dtype=np.float16)

    for epoch in range(start_epoch, max_epochs):
        last_time = time.time()
        train_err = 0.0
        train_samples = 0
        network.train()
        for sample_batch in train_data_loader:
            optimizer.zero_grad()

            batch_loss = 0
            for sample in sample_batch:
                batch_loss += calculate_sample_loss(sample)

            # Scale the loss and call backward()
            scaling_factor = batch_loss / len(sample_batch)
            scaler.scale(scaling_factor).backward()
            scaler.step(optimizer)
            scaler.update()

            # train_err += batch_loss.item()  # not sure why DJ put this here.
            train_err += batch_loss
            train_samples += len(sample_batch)

        train_err /= train_samples
        train_losses[epoch] = train_err.item()  # I added this arrays, to save and plot the loss curves.

        # Run validation samples
        network.eval()
        val_err = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for batch in val_data_loader:
                for val_sample in batch:
                    val_err += calculate_sample_loss(val_sample)
                    val_sample_count += 1

            val_err /= val_sample_count
            #  scheduler.step(val_err)
            val_losses[epoch] = val_err.item()  # I added this arrays, to save and plot the loss curves.

            print(f'Epoch {epoch}, train loss: {train_err:.4f}, val loss: {val_err:.4f}')
            print(f'Time taken = {time.time() - last_time:.2f} s', flush=True)
            last_time = time.time()

            if val_err < val_err_min:
                val_err_min = val_err
                # torch.save(network.state_dict(), Filename.prot_e2e_model_pt.value)
                # torch.save(network.state_dict(), 'prot_e2e_model.pt')
                _atomic_torch_save(data=network.state_dict(), pt_fname='prot_e2e_model.pt')

            # torch.save(network.state_dict(), Filename.prot_e2e_model_train_pt.value)
            # torch.save(network.state_dict(), 'prot_e2e_model_train.pt')
            _atomic_torch_save(data=network.state_dict(), pt_fname='prot_e2e_model_train.pt')

            _atomic_torch_save(data={'epoch': epoch, 'val_err_min': val_err_min}, pt_fname='checkpoint.pt')

        epochs[epoch] = epoch

    return epochs, train_losses, val_losses


if __name__ == "__main__":
    # print(f'os.getcwd()={os.getcwd()}')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # TO HELP DEBUGGING
    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # TO HELP DEBUGGING
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Debugging: CUDA is available. Running on:", device)
    else:
        print("Debugging: Not finding your CUDA... something is wrong !! ")

    check_runtime_specs = False
    if check_runtime_specs:
        import pandas
        import numpy
        import torch
        import einops

        # Print the active Conda environment (if any)
        print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV"))

        # Print the environment paths
        print("Environment PATH:", os.environ.get("PATH"))

        print(f'PyTorch CUDA version: {torch.version.cuda}')
        print(f'IS CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA devices: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                print(f'Device {i}: {torch.cuda.get_device_name(i)}')

        import subprocess
        # result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.run(['which', 'nvcc'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'result.stdout.decode(): {result.stdout.decode()}')

        print(f'torch.__version__={torch.__version__}')
        print(f'pandas.__version__={pandas.__version__}')
        print(f'numpy.__version__={numpy.__version__}')
        print(f'einops.__version__={einops.__version__}')
        print(f'torch.__version__={torch.__version__}')
        print(f'sys.version = {sys.version}')

    # SET UP LOSSES PATH AND CHECK OK *BEFORE* STARTING TRAINING:
    _abs_path = os.path.dirname(os.path.abspath(__file__))
    path_lpe_txt = '../losses/losses_per_epoch_16Dec.txt'
    path_lpe_txt = os.path.join(_abs_path, path_lpe_txt)
    path_lpe_txt = os.path.normpath(path_lpe_txt)
    path_lpe_dir = '../losses'
    path_lpe_dir = os.path.join(_abs_path, path_lpe_dir)
    path_lpe_dir = os.path.normpath(path_lpe_dir)
    assert os.path.exists(path_lpe_dir), ("Missing `losses` directory. Needed for saving loss per epoch data. "
                                          "It should be present in `src` directory at same level as `diffusion` dir.")

    # _targetfile_lst_path = Path.rp_diffdata_9_PDBids_lst.value
    lst_file = 'pdbchains_565.lst'
    _targetfile_lst_path = f'../diffusion/diff_data/PDBid_list/{lst_file}'
    _targetfile_lst_path = os.path.join(_abs_path, _targetfile_lst_path)
    assert os.path.exists(_targetfile_lst_path), f'{_targetfile_lst_path} cannot be found. Btw, cwd={os.getcwd()}'

    start_time = time.time()
    _epochs, _train_losses, _val_losses = main(_targetfile_lst_path)
    print(f"`main()` for 565 PDBs completed in {(time.time() - start_time) / 3600:.2f} hours", flush=True)

    losses_per_epoch = np.column_stack((_epochs, _train_losses, _val_losses))
    np.savetxt(path_lpe_txt, losses_per_epoch, fmt=('%d', '%.2f', '%.2f'), delimiter=',')
    loss_plotter.plot_train_val_errors_per_epoch(path_lpe_txt)
