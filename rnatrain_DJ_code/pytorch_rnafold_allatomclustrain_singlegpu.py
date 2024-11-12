#!/home/jones/miniconda3/bin/python

# Diffusion method for RNA folding

import sys
import os
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
from nndef_rnafold_atompyt2 import DiffusionNet

# Limit program to only see & use GPU with ID 0 (first GPU on system). Otherwise, program can access all available GPUs.
# Including this line would seem to restrict the effectiveness of the scheduler to spread workload for different users.
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# BATCH_SIZE = 32
BATCH_SIZE = 8
# NSAMPLES = 24
NSAMPLES = 12
SIGDATA = 16
        
RESTART_FLAG = True
FINETUNE_FLAG = False


# Load dataset function remains the same
def load_dataset():

    train_list = []
    validation_list = []
    tnum = 0

    atokendict = {"OP3": 0, "P": 1, "OP1": 2, "OP2": 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9,
                  "C2'": 10, "O2'": 11, "C1'": 12, "N9": 13, "C8": 14, "N7": 15, "C5": 16, "C6": 17, "O6": 18,
                  "N1": 19, "C2": 20, "N2": 21, "N3": 22, "C4": 23, "O2": 24, "N4": 25, "N6": 26, "O4": 27}

    ntnumdict = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    sum_d2 = 0
    sum_d = 0
    nn = 0
    
    with open('train_clusters.lst', 'r') as targetfile:
        for line in targetfile:
            targets = line.rstrip().split()
            sp = []
            for target in targets:
                ntcodes = []
                ntindices = []
                bbindices = []
                atomcodes = []
                coords = []
                ntindex = -1
                atomindex = 0
                lastnid = None

                with open('data/cif/' + target + '.cif', 'r') as pdbfile:
                    for line in pdbfile:
                        if line[:4] == 'ATOM':
                            fields = line.split()
                            #  0   1 2   3   4 5 6 7 8 9 10    11  12    13  14   15  16 17 18   19   20
                            # ATOM 1 O "O5'" . G A ? 1 ? 21.3 -9.4 18.8  1.0 90.2  ?  1  G   A  "O5'"  1
                            atom_id = fields[3]  # like O5 or C5 or C3 etc
                            nucleotide = fields[5]  # the RNA base (A,G, U or C)
                            nucleotide_index = fields[8]  # the nucleotide index
                            occupancy = fields[13]

                            # atid = fields[3].replace('"', '')
                            atid = atom_id.replace('"', '')
                            # if atid not in atokendict or float(fields[13]) <= 0.5:
                            if atid not in atokendict or float(occupancy) <= 0.5:
                                continue
                            # if fields[8] != lastnid:
                            if nucleotide_index != lastnid:  #
                                # nt = ntnumdict.get(fields[5], 4)
                                # nt = ntnumdict.get(nucleotide, 4)  # label_comp_id is the nt
                                # ntcodes.append(nt)
                                ntcodes.append(ntnumdict.get(nucleotide, 4))
                                bbindices.append(atomindex)
                                ntindex += 1
                                # lastnid = fields[8]
                                lastnid = nucleotide_index
                            if atid == "C3'" or atid == "P":
                                # Replace representative reference atom index with preferred type (C3' > P)
                                bbindices[-1] = atomindex
                            # Split the line
                            xyz_fields = [fields[10], fields[11], fields[12]]
                            coords.append(np.array([float(xyz_fields[0]), float(xyz_fields[1]), float(xyz_fields[2])]))
                            ntindices.append(ntindex)
                            atomcodes.append(atokendict[atid])
                            atomindex += 1
                length = ntindex + 1

                if length < 10 or length > 500:
                    continue

                pdb_embed = torch.load("data/emb/" + target + ".pt")
                assert pdb_embed.size(1) == length

                assert length == len(bbindices)

                if len(ntindices) < length * 6:
                    print("WARNING: Too many missing atoms in ", target, length, len(ntindices))
                    continue

                ntcodes = np.asarray(ntcodes, dtype=np.uint8)
                atomcodes = np.asarray(atomcodes, dtype=np.uint8)
                bbindices = np.asarray(bbindices, dtype=np.int16)
                ntindices = np.asarray(ntindices, dtype=np.int16)

                target_coords = np.asarray(coords, dtype=np.float32)
                target_coords -= target_coords.mean(0)

                assert length == target_coords[bbindices].shape[0]

                # For computing standard deviation
                sum_d2 += (target_coords ** 2).sum()
                sum_d += np.sqrt((target_coords ** 2).sum(axis=-1)).sum()
                nn += target_coords.shape[0]

                diff = target_coords[1:] - target_coords[:-1]
                distances = np.linalg.norm(diff, axis=1)
                # why is distances being calculated? What is diff doing ?
                # difference between all rows excluding row 0 and all rows excluding last row.
                print(target_coords.shape, target, length, distances.min(), distances.max())

                sp.append((ntcodes, atomcodes, ntindices, bbindices, target, target_coords))

            # Choose every 10th sample for validation
            if tnum % 10 == 0:
                validation_list.append(sp)
            else:
                train_list.append(sp)
            tnum += 1
        
    sigma_data = sqrt((sum_d2 / nn) - (sum_d / nn) ** 2)
    print("Data s.d. = ", sigma_data)
    print("Data unit var scaling = ", 1/sigma_data)

    return train_list, validation_list


# Superpose coordinates (superposes c1 on c2)
def lsq_fit(c1, c2):
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

        try:
            U, S, Vh = torch.linalg.svd(cov)
        except RuntimeError:
            return None

        V = Vh.transpose(-2, -1)
        d = torch.eye(3, device=P.device).repeat(P.size(0),1,1)
        d[:, 2, 2] = torch.det(torch.matmul(V, U.transpose(-2, -1)))

        rot = torch.matmul(torch.matmul(V, d), U.transpose(-2, -1))
        rot_P = torch.matmul(rot, P)

    return (rot_P + Q_mean).transpose(1, 2)


def random_rotation_matrices(N):
    """Generates N random rotation matrices."""
    axes = np.random.randn(N, 3)
    axes /= np.linalg.norm(axes, axis=1)[:, np.newaxis]  # normalize each vector
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
        #self.mse_sum = 0
        #self.mse_count = 0
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, tn):
        if self.augment:
            sample = random.choice(self.sample_list[tn])
        else:
            sample = self.sample_list[tn][0]
        ntseq = sample[0]
        atomcodes = sample[1]
        ntindices = sample[2]
        bbindices = sample[3]
        target = sample[4]
        target_coords = sample[5]

        embed = torch.load("data/emb/" + target + ".pt")
        
        length = ntseq.shape[0]

        if FINETUNE_FLAG:
            croplen = 20
        else:
            croplen = random.randint(10, min(20, length))

        # print(f'croplen={croplen}')

        if self.augment and length > croplen:
            lcut = random.randint(0, length-croplen)
            ntseq = ntseq[lcut:lcut+croplen]
            bbindices = bbindices[lcut:lcut+croplen]
            bb_coords = target_coords[bbindices]
            embed = embed[:,lcut:lcut+croplen]
            mask = np.logical_and(ntindices >= lcut, ntindices < lcut+croplen)
            atomcodes = atomcodes[mask]
            ntindices = ntindices[mask] - lcut
            target_coords = target_coords[mask]
            length = croplen
        else:
            bb_coords = target_coords[bbindices]

        if target_coords.shape[0] < 10:
            print(target, length, ntindices)
            
        noised_coords = target_coords - target_coords.mean(axis=0)

        # Original coordinates and replicating for N sets (N, L, 3)
        batched_coords = np.repeat(noised_coords[np.newaxis, :, :], NSAMPLES, axis=0)

        if self.augment:
            # Generate N rotation matrices (N, 3, 3)
            rotation_matrices = random_rotation_matrices(NSAMPLES)
            translations = np.random.randn(NSAMPLES,1,3)
            # Apply rotations using einsum for batch matrix multiplication
            batched_coords = np.einsum('nij,nkj->nki', rotation_matrices, batched_coords) + translations
            # distribution = torch.distributions.Beta(1, 8)
            distribution = torch.distributions.Uniform(0, 1)
            tsteps = distribution.sample((NSAMPLES,))
        else:
            tsteps = torch.arange(0, 1, 1 / NSAMPLES)

        sig_max_r7 = (SIGDATA * 10) ** (1/7)
        sig_min_r7 = 4e-4 ** (1/7)
        noise_levels = (sig_max_r7 + tsteps * (sig_min_r7 - sig_max_r7)) ** 7

        ntcodes = torch.from_numpy(ntseq.copy()).long()
        bb_coords = torch.from_numpy(bb_coords).float()
        ntindices = torch.from_numpy(ntindices.copy()).long()
        atomcodes = torch.from_numpy(atomcodes.copy()).long()
        target_coords = torch.from_numpy(target_coords.copy()).unsqueeze(0)

        batched_coords = torch.from_numpy(batched_coords).float()
        
        noise = torch.randn_like(batched_coords)

        # print(noise_levels.size(), noise.size(), batched_coords.size())
        noised_coords = noise_levels.view(NSAMPLES, 1, 1) * noise + batched_coords

        sample = (embed, noised_coords, noise_levels, noise, ntcodes, atomcodes, ntindices, bb_coords, target_coords,
                  target)

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
            # What is this rna_e2e_model_train.pt?
            # Is it an empty network ? Untrained weights or has it been trained on something?
            pretrained_dict = torch.load('rna_e2e_model_train.pt', map_location='cuda')
            model_dict = network.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            network.load_state_dict(pretrained_dict, strict=False)
        except:
            pass

        try:
            checkpoint = torch.load('checkpoint.pt')
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
        inputs = sample[0].cuda(non_blocking=True)
        noised_coords = sample[1].cuda(non_blocking=True)
        noise_levels = sample[2].cuda(non_blocking=True)
        ntcodes = sample[4].cuda(non_blocking=True)
        atomcodes = sample[5].cuda(non_blocking=True)
        ntindices = sample[6].cuda(non_blocking=True)
        bb_coords = sample[7].cuda(non_blocking=True)
        target_coords = sample[8].cuda(non_blocking=True)

        pred_denoised, pred_coords, pred_confs = network(inputs, ntcodes, atomcodes, ntindices, noised_coords, noise_levels)

        predmap = torch.cdist(pred_coords, pred_coords)
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
                torch.save(network.state_dict(), 'rna_e2e_model.pt')
                print("Saving model...", flush=True)
                    
            torch.save(network.state_dict(), 'rna_e2e_model_train.pt')

            torch.save({
                'epoch': epoch,
                'val_err_min': val_err_min,
            }, 'checkpoint.pt')


if __name__ == "__main__":
    import pandas
    import numpy
    import torch
    import einops

    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print(f'IS CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'Device {i}: {torch.cuda.get_device_name(i)}')

    # import subprocess
    # result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f'result.stdout.decode(): {result.stdout.decode()}')

    print(f'torch.__version__={torch.__version__}')
    print(f'pandas.__version__={pandas.__version__}')
    print(f'numpy.__version__={numpy.__version__}')
    print(f'einops.__version__={einops.__version__}')
    print(f'torch.__version__={torch.__version__}')
    print(f'sys.version = {sys.version}')

    main()
