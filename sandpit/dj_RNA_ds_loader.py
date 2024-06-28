# DJ example file

import numpy as np
import torch


def load_dataset():

    train_list = []
    validation_list = []
    tnum = 0

    atokendict = {"OP3": 0, "P": 1, "OP1": 2, "OP2": 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8,
                  "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12, "N9": 13, "C8": 14, "N7": 15, "C5": 16, "C6": 17,
                  "O6": 18, "N1": 19, "C2": 20, "N2": 21, "N3": 22, "C4": 23, "O2": 24, "N4": 25, "N6": 26, "O4": 27}

    ntnumdict = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    sum_d2 = 0
    sum_d = 0
    nn = 0

    # with open('rnacif_train.lst', 'r') as targetfile:
    with open('../data/rnacif_train.lst', 'r') as targetfile:

        for line in targetfile:
            target = line.rstrip()
            # target = ''
            ntcodes = []
            ntindices = []
            bbindices = []
            atomcodes = []
            coords = []
            ntindex = -1
            atomindex = 0
            lastnid = None

            with open('../data/' + target + '.cif', 'r') as pdbfile:
            # with open('data/cif/' + target + '.cif', 'r') as pdbfile:
            # with open('../data/' + target + '4gxy.cif', 'r') as pdbfile:
                for line in pdbfile:
                    if line[:4] == 'ATOM':
                        fields = line.split()
                        atid = fields[3].replace('"', '')
                        if atid not in atokendict or float(fields[13]) <= 0.5:
                            continue
                        if fields[8] != lastnid:
                            nt = ntnumdict.get(fields[5], 4)
                            ntcodes.append(nt)
                            bbindices.append(atomindex)
                            ntindex += 1
                            lastnid = fields[8]
                        if atid == "C3'" or atid == "P":
                            # Replace representative reference atom index with preferred type (C3' > P)
                            bbindices[-1] = atomindex
                        # Split the line
                        xyz_fields = [fields[10], fields[11], fields[12]]
                        coords.append(np.array([float(xyz_fields[0]), float(xyz_fields[1]), float(xyz_fields[2])]))
                        ntindices.append(ntindex)
                        atomcodes.append(atokendict[atid])
                        atomindex += 1

            length = ntindex+1

            if length < 10 or length > 500:
                continue

            # print(torch.load("data/emb2/" + target + ".pt"))
            assert torch.load("data/emb2/" + target + ".pt").size(1) == length

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

            sum_d2 += (target_coords ** 2).sum()
            sum_d += np.sqrt((target_coords ** 2).sum(axis=-1)).sum()
            nn += target_coords.shape[0]

            diff = target_coords[1:] - target_coords[:-1]
            distances = np.linalg.norm(diff, axis=1)

            print(target_coords.shape, target, length, distances.min(), distances.max())

            sp = (ntcodes, atomcodes, ntindices, bbindices, target, target_coords)

            if tnum % 50 == 0:
                validation_list.append(sp)
            else:
                train_list.append(sp)
            tnum += 1

    sigma_data = np.sqrt((sum_d2 / nn) - (sum_d / nn) ** 2)
    print("Data s.d. = ", sigma_data)
    print("Data unit var scaling = ", 1/sigma_data)

    return train_list, validation_list


if __name__ == '__main__':
    load_dataset()
