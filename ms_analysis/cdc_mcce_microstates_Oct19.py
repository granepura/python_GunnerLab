import os
import numpy as np
import gromacs

gromacs.config.setup()
from pymembrane.util.CDC.run_cdc import *
from matplotlib.pyplot import rcParams

dict_cofactors = {'SQD': dict_compelete_SQD_mcce,
                  'HOH': dict_HOH_semiempirical,
                  'BCR': dict_BCR_antechamber,
                  }

# path to the mc pdbs dir:
# dir = 'PATH_TO_YOUR_MS_PDB_DIRECTORY_WHERE_YOU_WILL_RUN_CDC_CALCULATION'
path = os.getcwd()
dir = f'{path}/parsed_pdb_output_mc/'
#dir = f'{path}/test_parsed/'

# Create a dictionary to store site energies for each microstate
dict_site_energy = {}

for filename in os.listdir(dir):
    print(filename)
    if not os.path.isdir(os.path.join(dir, filename)) and filename != '.DS_Store':
        if filename.endswith('.pdb'):
            #print(f'{filename=}')
            site_energy_list = []
            list_total_contribution = []
            pdb_path = os.path.join(dir, filename)
            pdb_name = filename.split('.')[0]
            md_scratch_dir = f'{dir}CDC_{pdb_name}/'
            #print(md_scratch_dir)
            extended_pdb_name = f'extended_{pdb_name}.pdb'
            list_ligands = ['CLA', 'BCR', 'SQD', 'HOH']
            md_object = DynamicStructure(path_scratch=md_scratch_dir)
            # md_object.prepare_minimization(pdb_path=pdb_path,
            #                                protein_force_field='amber99',
            #                                list_ligands=list_ligands,
            #                                dict_update_residue={'D0003': 'ASP',
            #                                                     'D0179': 'ASP',
            #                                                     'C0179': 'ASP',
            #                                                     'C0003': 'ASP',
            #                                                     'B0179': 'ASP',
            #                                                     'B0004': 'GLU',
            #                                                     'A0179': 'ASP'},
            #                                water_forceField='tip4p',
            #                                ignh=False,
            #                                minimize_ligand=False,
            #                                run_minimization=False
            #                                )

            # md_object.build_extended_pdb(dict_cofactors)
            protein_atomic = ProteinAtomic(f'{dir}{filename}', "Isia", center_atomic=False, set_atomic_charges=True)
            protein_atomic.prepare_pigments('CLA', ChlorophyllAtomic,
                                            q_0011_charges_dict='CLA_mcce',
                                            qy_atoms=('N1B', 'N1D'),
                                            qx_atoms=('N1A', 'N1C'))

            dict_site_energy_shift_mcce, dict_total_contribution_mcce = calculate_cdc_site_shift(
                protein_atomic, dielectric_eff=2.5, protein_contribution=True)
            dict_site_energy_mcc = calculate_total_site_energy(dict_site_energy_shift_mcce, 14950, 15674)

            md_object.create_dir('cdc_results')
            np.save(f'{md_scratch_dir}cdc_results/dict_total_contribution_mcce.npy', dict_total_contribution_mcce)

            # Save the site energy dictionary for each microstate
            dict_site_energy[pdb_name] = dict_site_energy_shift_mcce
            print(dict_site_energy_shift_mcce)
            md_object.save_dict_2_txt(dict_data=dict_site_energy_shift_mcce,
                                      text_file_name=f'{md_scratch_dir}cdc_results/dict_site_energy_shift_most_occ_'
                                                     f'{list_ligands=}')

            md_object.save_dict_2_txt(dict_data=dict_site_energy_mcc,
                                      text_file_name=f'{md_scratch_dir}cdc_results/dict_total_site_energy_most_occ_'
                                                     f'{list_ligands=}')

# Save the site energy dictionary for all microstates
np.save(f'{dir}dict_site_energies.npy', dict_site_energy)
