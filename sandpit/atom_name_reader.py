from Bio.PDB.MMCIFParser import MMCIFParser
import warnings  # Load the Chemical Component Dictionary CIF file
import gemmi
import json  # Function to extract atom names for each amino acid residue

if __name__ == '__main__':
    # Function to extract atom names for each amino acid residue
    def extract_atom_names_per_residue(cif_file_path):
        print('here')
        pass
        doc = gemmi.cif.read_file(cif_file_path)
        atom_dict = {}  # Loop through each data block in the CIF file
        for block in doc:
            # Loop through each row in the '_chem_comp_atom' loop
            if '_chem_comp_atom' in block:
                for item in block.find('_chem_comp_atom'):
                    residue_name = item['_chem_comp_atom.comp_id']
                    atom_name = item['_chem_comp_atom.atom_id']
                    if residue_name not in atom_dict:
                        atom_dict[residue_name] = set()
                    atom_dict[residue_name].add(atom_name)  # Convert sets to lists for JSON serialization
        for residue_name in atom_dict:
            atom_dict[residue_name] = list(atom_dict[residue_name])
            return atom_dict  # Path to the locally downloaded CIF file


    cif_file_path = '../data/big_files_to_git_ignore/components.cif'  # Extract atom names per residue
    atom_dict = extract_atom_names_per_residue(cif_file_path)  # Write the dictionary to a JSON file
    with open('atom_names_per_residue.json', 'w') as json_file:
        json.dump(atom_dict, json_file, indent=4)
        print("Atom names per residue have been written to 'atom_names_per_residue.json'.")

