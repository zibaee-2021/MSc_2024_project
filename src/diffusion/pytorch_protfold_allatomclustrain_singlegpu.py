#!~/miniconda3/bin/python
"""
Training script for protein structure diffusion model
(deliberately made as few changes as possible from original RNA script).

"""
import os
from typing import List, Tuple
from numpy.typing import NDArray
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from nndef_protfold_atompyt2 import DiffusionNet
import dataset_loader as dsl
from src.losses import loss_plotter


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
BATCH_SIZE = 8
NSAMPLES = 12
SIGDATA = 16

RESTART_FLAG = False  # False to build first train model
FINETUNE_FLAG = False


def _atomic_torch_save_pt(data: dict, pt_fname: str) -> None:
    assert data is not None, 'The model dict or checkpoint dict passed in to _atomic_torch_save() is None'
    assert bool(data), 'The model dict or checkpoint dict passed in to _atomic_torch_save() is empty'
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_temp = f"pt_files/{pt_fname.removesuffix('.pt') + '.tmp'}"
    abspath_temp = os.path.normpath(os.path.join(abs_path, path_temp))
    os.makedirs(os.path.dirname(abspath_temp), exist_ok=True)
    torch.save(data, abspath_temp)
    assert os.path.exists(abspath_temp), f"{pt_fname.removesuffix('.pt') + '.tmp'} was not created at '{abspath_temp}'."
    path_pt = f'pt_files/{pt_fname}'
    abspath_pt = os.path.normpath(os.path.join(abs_path, path_pt))
    os.replace(abspath_temp, abspath_pt)
    assert os.path.exists(abspath_pt), f"'{abspath_temp}' was not renamed to '{abspath_pt}'."
    print(f"Model '{pt_fname}' saved.", flush=True)


def _torch_load_pt(pt_fname: str, is_embedding_file=False):
    """
    Load torch data that is in the form of a `.pt` file.
    :param pt_fname: Filename of `.pt` file to be loaded.
    :param is_embedding_file: True if loading an embedding, which will therefore be in `diff_data/emb` directory.
    Otherwise, load from `pt_files` dir.
    :return: A Tensor or a dict.
    """
    abs_path = os.path.dirname(os.path.abspath(__file__))
    pt_fname = pt_fname.removesuffix('.pt')
    if is_embedding_file:
        path_pt_fname = f'diff_data/emb/{pt_fname}.pt'
    else:
        path_pt_fname = f'pt_files/{pt_fname}.pt'
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
    # Note: I do not check other types than dict and Tensor.
    return pt


def lsq_fit(c1, c2):
    """
    Superpose given coordinates.
    To compare two sets of 3D coordinates in a way that is rotation-invariant, this function performs the rotation
    that minimises any rotational differences, so that subsequent comparisons are a rigid transformation fitting,
    using least squares method for minimising the distance between two sets of coordinates. Minimisation is achieved
    using SVD. (This involves aligning points, via rotation, rather than fitting a model to data as in the application
    of least squares that I am more used to encountering). Note the sum of squared errors does not need to be
    explicitly calculated because the use of SVD shortcuts to the optimal rotation matrix, so its performed implicitly.
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
    """
    Generate N random 3D rotation matrices for data augmentation purposes.
    """
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
    """
    Dataset class for Diffusion Modelling Pipeline (DMP).
    Encapsulate training and validation datasets in PyTorch Datasets (`torch.utils.data.Dataset`).
    Additionally, handle data augmentation with random peptide fragments, randomly rotated and noised.
    (These datasets will be passed as inputs to PyTorch DataLoaders, which will then call __getitem__() to extract
    the sample: (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords,
    target))
    """

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
        aacodes = sample[0]
        atomcodes = sample[1]
        aaindices = sample[2]
        bbindices = sample[3]
        target = sample[4]
        target_coords = sample[5]

        embed = _torch_load_pt(pt_fname=target, is_embedding_file=True)

        linear_layer = nn.Linear(768, 1024, bias=False)  # to match up the dimensions
        embed = linear_layer(embed)
        embed = embed.detach()

        prot_length = aacodes.shape[0]

        if FINETUNE_FLAG:
            peptide_window = 20
        else:
            peptide_window = random.randint(10, min(20, prot_length))
        # print(f'peptide_window={peptide_window}')

        if self.augment and prot_length > peptide_window:
            start = random.randint(0, prot_length - peptide_window)
            end = start + peptide_window
            aacodes = aacodes[start: end]
            bbindices = bbindices[start: end]
            bb_coords = target_coords[bbindices]  #
            embed = embed[:, start: end]
            mask = np.logical_and(aaindices >= start, aaindices < end)
            atomcodes = atomcodes[mask]
            aaindices = aaindices[mask] - start
            target_coords = target_coords[mask]
            prot_length = peptide_window

        else:
            bb_coords = target_coords[bbindices]

        if target_coords.shape[0] < 10:
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

        aacodes = torch.from_numpy(aacodes.copy()).long()
        bb_coords = torch.from_numpy(bb_coords).float()
        aaindices = torch.from_numpy(aaindices.copy()).long()
        atomcodes = torch.from_numpy(atomcodes.copy()).long()
        target_coords = torch.from_numpy(target_coords.copy()).unsqueeze(0)
        batched_coords = torch.from_numpy(batched_coords).float()
        noise = torch.randn_like(batched_coords)
        # print(noise_levels.size(), noise.size(), batched_coords.size())
        noised_coords = noise_levels.view(NSAMPLES, 1, 1) * noise + batched_coords
        sample = (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords, target)
        return sample


def main() -> Tuple[NDArray[np.int16], NDArray[np.float16], NDArray[np.float16]]:

    if not RESTART_FLAG:
        print(f"`RESTART_FLAG` is False, hence don't try to use the pre-built models: 'prot_e2e_model_train.pt' and "
              f"'checkpoint.pt'")
    else:
        assert os.path.exists('prot_e2e_model_train.pt'), ("Expected 'prot_e2e_model_train.pt' model file is absent. "
                                                           "Cannot proceed until this is addressed.")
    print(f'FINETUNE_FLAG={FINETUNE_FLAG}')

    global BATCH_SIZE
    
    network = DiffusionNet(seqwidth=1024,
                           atomwidth=128,
                           seqheads=16,
                           atomheads=8,
                           seqdepth=6,
                           atomdepth=3,
                           cycles=2).cuda()

    print("Loading data...")
    train_list, validation_list = dsl.load_dataset()

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
            pt_fname = 'prot_e2e_model_train.pt'
            pretrained_dict = _torch_load_pt(pt_fname=pt_fname)
            model_dict = network.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if (k in model_dict)
                               and (model_dict[k].shape == pretrained_dict[k].shape)}
            network.load_state_dict(pretrained_dict, strict=False)
        except:
            print("torch.load() of pretrained 'prot_e2e_model_train.pt' model went ok, but subsequent "
                  "network.state_dict() or network.load_state_dict() seems to have a problem.")
            pass

        try:
            checkpoint = _torch_load_pt(pt_fname='checkpoint.pt')
            start_epoch = checkpoint['epoch']
            val_err_min = checkpoint['val_err_min']
        except:
            print("torch.load() of pretrained 'checkpoint.pt' went ok, but subsequent access to the dict's key-values"
                  "seems to have a problem.")
            pass

    optimizer = torch.optim.RAdam(network.parameters(), lr=max_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=10)
    # Initialise GradScaler to allow float16 processing
    scaler = GradScaler()
        
    print('Starting training...', flush=True)

    def calculate_sample_loss(_sample: List[torch.Tensor]) -> torch.FloatTensor:
        """
        Calculate loss for a single sample, combining 3 difference loss components in a weighted sum of backbone loss,
        confidence loss and difference loss.
        :param _sample: Tuple of the following 10 elements: `embed`, `noised_coords`, `noise_levels`, `noise`,
        `aacodes`, `atomcodes`, `aaindices`, `bb_coords`, `target_coords` and `target`. All of which are used except
        for `noise` (sample[3]) and `target` (sample[9]).
        :return: the combined loss.
        """
        # not sure about `non_blocking` argument. Does it reduce computation time, in asynchronous context?
        # (embed, noised_coords, noise_levels, noise, aacodes, atomcodes, aaindices, bb_coords, target_coords, target)
        inputs = _sample[0].cuda(non_blocking=True)  # pLM embedding tensor, `embed`.
        noised_coords = _sample[1].cuda(non_blocking=True)  # prediction after noise added (tbc)
        noise_levels = _sample[2].cuda(non_blocking=True)  # how much noise to add. (Constant defined in DMPDataset))
        aacodes = _sample[4].cuda(non_blocking=True)  # Enumeration of amino acids (0-19)
        atomcodes = _sample[5].cuda(non_blocking=True)  # Enumeration of atoms (0-37 or 0-186)
        aaindices = _sample[6].cuda(non_blocking=True)  # zero-indexed position of amino acid in protein.
        bb_coords = _sample[7].cuda(non_blocking=True)  # mean-corrected  X,Y,Z coords of anchor backbone atoms.
        target_coords = _sample[8].cuda(non_blocking=True)  # mean-corrected X,Y,Z coords of atoms.
        print(f'Calc loss PDBid={_sample[9]}')

        # PASS THROUGH NEURAL NETWORK TO COMPUTE PREDICTIONS OF THE DENOISED STRUCTURE, COORDINATES AND
        # PER-RESIDUE CONFIDENCES
        # pred_denoised, pred_coords, pred_confs = network(x=inputs, aacodes=aacodes, atcodes=atomcodes,
        #                                                  aaindices=aaindices, noised_coords=noised_coords,
        #                                                  nlev_in=noise_levels)
        pred_denoised, pred_coords, pred_confs = network(inputs, aacodes, atomcodes, aaindices, noised_coords, noise_levels)

        # WEIGHTED SUM OF THREE LOSS CALCULATIONS:---------------------------------------------------------------------
        # 1. CALCULATE BACKBONE LOSS BY MSE BETWEEN PAIRWISE DISTANCES:
        predmap = torch.cdist(pred_coords, pred_coords)  # COMPUTE PAIRWISE EUCLID DISTS TWEEN ALL PREDICTED ATOM COORDS
        bb_coords = bb_coords.unsqueeze(0)  # ADD DIMENSION TO MATCH: CHANGING FROM (N, D) to (1, N, D).
        targmap = torch.cdist(bb_coords, bb_coords)  # COMPUTE PAIRWISE EUCLID DISTS TWEEN ALL GROUND TRUTH BB COORDS
        bb_loss = F.mse_loss(predmap, targmap)  # ELEMENT-WISE MEAN-SQUARED ERROR

        # 2. COMPUTE CONFIDENCE LOSS BY DIFFERENCE BETWEEN PAIRWISE DISTANCES PREDICTED CONFIDENCE SCORES AND LDDT:
        diffmap = (targmap - predmap).abs().squeeze(0)  # DIFFERENCE BETWEEN PREDICTED PW DISTANCES AND TARGET PW DISTS.
        incmask = (targmap < 15.0).float().squeeze(0).fill_diagonal_(0)  # FOR CALCULATING LDDT.
        lddt = (0.25 * ((diffmap < 0.5).float() +
                        (diffmap < 1.0).float() +
                        (diffmap < 2.0).float() +
                        (diffmap < 4.0).float()) * incmask).sum(dim=0) / torch.clip(incmask.sum(dim=0), min=1)
        conf_loss = (torch.sigmoid(pred_confs).squeeze(0) - lddt).abs().mean()  # DIFF TWEEN PREDICTED CONF AND LDDT.

        # 3. COMPUTE DIFFUSION LOSS AS DIFFERENCE BETWEEN PREDICTED COORDS AND TARGET COORDS (ROTATED TO BEST MATCH
        # PREDICTED COORDS:
        rot_targets = lsq_fit(target_coords, pred_denoised)  # Rotate target to best align the 3d coords of predicted.
        nwts = (noise_levels.pow(2) + SIGDATA * SIGDATA) / (noise_levels + SIGDATA).pow(2)  # noise weighting score
        diff_loss = ((rot_targets - pred_denoised).pow(2).sum(dim=2).mean(dim=1) * nwts).mean()

        # COMPUTE TOTAL LOSS AS WEIGHTED SUM OF BACKBONE, CONFIDENCE AND DIFFUSION LOSSES:
        loss = bb_loss + 0.01 * conf_loss + diff_loss

        # If loss is null just return zero
        if loss != loss:  # (logic unclear to me here)
            loss = 0

        return loss

    # NP ARRAYS TO SAVE LOSSES FOR LOSS CURVES:
    epochs = np.zeros(max_epochs, dtype=np.int16)
    train_losses = np.zeros(max_epochs, dtype=np.float16)
    val_losses = np.zeros(max_epochs, dtype=np.float16)

    # TRAIN:
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

            scaling_factor = batch_loss / len(sample_batch)
            scaler.scale(scaling_factor).backward()
            scaler.step(optimizer)
            scaler.update()

            train_err += batch_loss
            train_samples += len(sample_batch)

        train_err /= train_samples
        train_losses[epoch] = train_err.item()

        # VALIDATION:
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
            last_time = time.time()  # <- already done on line 413 at start of epoch

            # SAVE WEIGHTS:
            if val_err < val_err_min:
                val_err_min = val_err
                _atomic_torch_save_pt(data=network.state_dict(), pt_fname='prot_e2e_model.pt')

            _atomic_torch_save_pt(data=network.state_dict(), pt_fname='prot_e2e_model_train.pt')
            _atomic_torch_save_pt(data={'epoch': epoch, 'val_err_min': val_err_min}, pt_fname='checkpoint.pt')

        epochs[epoch] = epoch

    return epochs, train_losses, val_losses


if __name__ == "__main__":

    # CHECK FOR CUDA:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Debugging: CUDA is available. Running on:", device)
    else:
        print("Debugging: No CUDA found... script will crash..")

    # ASSERT PATH TO DESTINATION OF LOSSES FILES *BEFORE* START OF TRAINING:
    _abs_path = os.path.dirname(os.path.abspath(__file__))

    import datetime
    dateMonth = datetime.datetime.now().strftime("%d%b")  # e.g. '20Jan'
    path_lpe_txt = os.path.normpath(os.path.join(_abs_path, f'../losses/losses_per_epoch_{dateMonth}.txt'))
    path_lpe_dir = os.path.normpath(os.path.join(_abs_path, '../losses'))
    assert os.path.exists(path_lpe_dir), ("Missing `losses` directory. Needed for saving loss per epoch data. "
                                          "It should be present in `src` directory at same level as `diffusion` dir.")

    # TRAIN MODEL:
    start_time = time.time()

    _epochs, _train_losses, _val_losses = main()
    print('Training completed.')
    time_taken = time.time() - start_time

    from pathlib import Path
    abspath_tok = os.path.normpath(os.path.join(_abs_path, 'diff_data/tokenised'))
    path = Path(abspath_tok)
    ssv_count = sum(1 for file in path.rglob("*.ssv"))
    print(f"Training on {ssv_count} PDBs completed in {time_taken / 3600:.2f} hours", flush=True)

    print(f'Saving losses to {path_lpe_txt}.')
    losses_per_epoch = np.column_stack((_epochs, _train_losses, _val_losses))
    np.savetxt(path_lpe_txt, losses_per_epoch, fmt=('%d', '%.2f', '%.2f'), delimiter=',')

    # Not sure this will plot if running from terminal:
    # loss_plotter.plot_train_val_errors_per_epoch(path_lpe_txt, include_train=True, include_val=True)

