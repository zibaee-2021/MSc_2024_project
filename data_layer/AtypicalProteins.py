# This Enum of atypical outlier proteins is not being used yet. Maybe after standard ones are used.

from enum import Enum


class UniprotId(Enum):
    Myoglobin_pig = 'P02189'  # PDB '1mwd' pig WT deoxy myoglobin
    # Myoglobin_sperm_whale = 'P02185'  # PDB '104M'
    Serine_Protease_Human = 'Q6UWY2'  # Serine protease 57
    Low_resolution_structure = ''  # resolution > 4 angstroms ??
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)


class PDBids(Enum):
    Myoglobin_pig = '1mwd'  # WT deoxy myoglobin
    Serine_Protease_Human = '4Q7Z' # 'Q6UWY2' X-ray 1.40 Ang aa34-283
    Low_resolution_structure = ''  # resolution > 4 angstroms ??
    Cyclic_peptide = ''
    NMR_structure = ''  # for fun ! (you'd need to skip all the Hydrogens)
