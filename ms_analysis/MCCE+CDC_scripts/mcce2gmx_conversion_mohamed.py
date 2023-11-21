import numpy as np
import shutil
from pymembrane.util.mcce_util.MCCEtoGromacsConverter import *


# path = '/Users/mohamedelrefaiy/Library/CloudStorage/Box-Box/Reseach/projects/2023/ISIA/MCCE/Aug_27_2023/trimer/'
path = os.getcwd()
ms_mcce_pdb_dir = f'{path}/pdb_output_mc/'        # path to the input dir
#ms_mcce_pdb_dir = f'{path}/ms_out/pH7eH0ms/pdbs_from_ms/'        # path to the input dir
path_dir_save = f'{path}/parsed_pdb_output_mc/'   # path to the output dir

if os.path.exists(path_dir_save):
    shutil.rmtree(path_dir_save)
    print(f'The directory {path_dir_save} has been deleted.')
os.mkdir(path_dir_save)


list_all_ligands = ['CLA', 'CLB', 'BCR', 'SQD', 'HOH', 'MEM']  # Should be updated based on the cofactors in the pdb

## The first key is the residue name found in the mcce input 'HIS', then 01 is the conformer name inside the mcce
# input, then 'HIP' is the residue name in amber force field.
dict_resname_mapping = {'HIS': {'02': 'HID', '01': 'HIE', '+1': 'HIP'},
                         'HIL': {'01': 'HID'},
                         'ASP': {'01': 'ASH', '02': 'ASH', '-1': 'ASP'},
                         'GLU': {'01': 'GLH', '02': 'GLH', '-1': 'GLU'},
                         'LYS': {'01': 'LYN'},
                         'CYS': {'-1': 'CYM'},
                         'CYX': {'01': 'CYD'},
                         'CYD': {'01': 'CYS'}}

for filename in os.listdir(ms_mcce_pdb_dir):
     if filename != '.DS_Store':
        print(filename)
        converter = MCCEtoGromacsConverter(mcce_pdb_path=f'{ms_mcce_pdb_dir}{filename}',
                                           list_all_ligands=list_all_ligands,
                                           dict_mapping_resname=dict_resname_mapping,
                                           saving_dir=path_dir_save)
        # converter.convert_mcce_to_gromacs(dict_update_protonation_residue={'B0179': 'ASP', 'C0179': 'KKK', 'D0003': 'ASP'})
        converter.convert_mcce_to_gromacs(pdb_output_name=f'{filename.split(".")[0]}_gromacs.pdb')

