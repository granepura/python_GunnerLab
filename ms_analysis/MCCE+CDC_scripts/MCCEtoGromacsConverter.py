import os
import numpy as np
import tempfile
from Bio.PDB import PDBParser, PDBIO, Select


amber_atomname_conversion_dict = {
            'GLY': {'HA2': 'HA1', 'HA3': 'HA2'},
            'SER': {'HB2': 'HB1', 'HB3': 'HB2'},
            'LEU': {'HB2': 'HB1', 'HB3': 'HB2'},
            'ILE': {'HG12': 'HG11', 'HG13': 'HG12', 'CD1': 'CD', 'HD11': 'HD1', 'HD12': 'HD2', 'HD13': 'HD3'},
            'ASN': {'HB2': 'HB1', 'HB3': 'HB2'},
            'GLN': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2'},
            'ARG': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2', 'HD2': 'HG1', 'HD3': 'HD2'},
            'HID': {'HB2': 'HB1', 'HB3': 'HB2'},
            'HIE': {'HB2': 'HB1', 'HB3': 'HB2'},
            'HIP': {'HB2': 'HB1', 'HB3': 'HB2'},
            'TRP': {'HB2': 'HB1', 'HB3': 'HB2'},
            'PHE': {'HB2': 'HB1', 'HB3': 'HB2'},
            'TYR': {'HB2': 'HB1', 'HB3': 'HB2'},
            'GLU': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2'},
            'ASP': {'HB2': 'HB1', 'HB3': 'HB2'},
            'LYS': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2', 'HD2': 'HD1', 'HD3': 'HD2',
                    'HE2': 'HE1', 'HE3': 'HE2'},
            'LYN': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2', 'HD2': 'HD1', 'HD3': 'HD2',
                    'HE2': 'HE1', 'HE3': 'HE2'},
            'PRO': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2', 'HD2': 'HD1', 'HD3': 'HD2'},
            'CYS': {'HB2': 'HB1', 'HB3': 'HB2'},
            'CYM': {'HB2': 'HB1', 'HB3': 'HB2'},
            'MET': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2'},
            'ASH': {'HB2': 'HB1', 'HB3': 'HB2', 'HD1': 'HD2'},
            'GLH': {'HB2': 'HB1', 'HB3': 'HB2', 'HG2': 'HG1', 'HG3': 'HG2', 'HE1': 'HE2'}
        }

dict_resname_mapping = {'HIS': {'02': 'HID', '01': 'HIE', '+1': 'HIP'},
                         'HIL': {'01': 'HID'},
                         'ASP': {'01': 'ASH', '02': 'ASH', '-1': 'ASP'},
                         'GLU': {'01': 'GLH', '02': 'GLH', '-1': 'GLU'},
                         'LYS': {'01': 'LYN'},
                         'CYS': {'-1': 'CYM'},
                         'CYX': {'01': 'CYD'},
                         'CYD': {'01': 'CYS'}}


class MCCEtoGromacsConverter:
    def __init__(self, mcce_pdb_path, list_all_ligands, dict_mapping_resname, saving_dir=None):
        self.mcce_pdb_path = mcce_pdb_path
        self.all_ligands = list_all_ligands     # list defining all the ligands in the mcce input file
        # self.list_2_keep = list_ligand_2keep    # list of all the ligands that should be only kept in the output PDB file.
        self.dict_mapping = dict_mapping_resname
        if saving_dir == None:
            self.saving_dir = os.path.dirname(self.mcce_pdb_path)
            print(f'when its none:{self.saving_dir=}')
        else:
            self.saving_dir = saving_dir
            print(f'when :{self.saving_dir=}')

        self.build_gromacs_dir()


    def delete_intermediate_files(self, file_list):
        for file_path in file_list:
            if os.path.exists(file_path):
                os.remove(file_path)

    def build_gromacs_dir(self):
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)


    def delete_intermediate_files(self, file_list):
        """
        Delete the intermediate extended PDBs during the conversion steps.
        """
        for file_path in file_list:
            if os.path.exists(file_path):
                os.remove(file_path)

    # def convert_mcce_to_gromacs(self):
    #
    #     # 1. Convert mcce PDB format to extended PDB format:
    #     extended_pdb_1 = os.path.join(self.saving_dir, "extended_pdb1.pdb")
    #     self.convert_to_extended_pdb(self.mcce_pdb_path, extended_pdb_1)
    #
    #     # 2. Rename the N and C terminal caps that mcce assigns to the terminal residues:
    #     extended_pdb_2 = os.path.join(self.saving_dir, "extended_pdb2.pdb")
    #     self.rename_capping_residues(extended_pdb_1, extended_pdb_2)
    #
    #     # 3. Assign the protonation states:
    #     extended_pdb_3 = os.path.join(self.saving_dir, "extended_pdb3.pdb")
    #     self.rename_residue_side_chains(extended_pdb_2, extended_pdb_3, self.dict_mapping)
    #
    #     # 4. Rename bk atoms:
    #     extended_pdb_4 = os.path.join(self.saving_dir, "extended_pdb4.pdb")
    #     self.rename_bk_atoms(extended_pdb_3, extended_pdb_4)
    #
    #     # 5. Rename atom names to Amber format:
    #     extended_pdb_5 = os.path.join(self.saving_dir, "extended_pdb5.pdb")
    #     self.rename_atomname2amber(extended_pdb_4, extended_pdb_5)
    #
    #     # 6. Keep protein residues and selected ligands
    #     final_output_file = os.path.join(self.saving_dir, "gromacs_most_occ_Mo.pdb")
    #     self.keep_protein_and_selected_ligands(
    #         extended_pdb_5, final_output_file, self.all_ligands
    #     )
    #
    #     # Delete intermediate files
    #     file_list = [extended_pdb_1, extended_pdb_2, extended_pdb_3, extended_pdb_4, extended_pdb_5]
    #     self.delete_intermediate_files(file_list)
    #     print(f'Conversionn is done!')

    def convert_mcce_to_gromacs(self, dict_update_protonation_residue=None, pdb_output_name='gromacs_most_occ_Mo.pdb'):
        # 1. Convert mcce PDB format to extended PDB format:
        extended_pdb_1 = os.path.join(self.saving_dir, "extended_pdb1.pdb")
        self.convert_to_extended_pdb(self.mcce_pdb_path, extended_pdb_1)

        # 2. Rename the N and C terminal caps that mcce assigns to the terminal residues:
        extended_pdb_2 = os.path.join(self.saving_dir, "extended_pdb2.pdb")
        self.rename_capping_residues(extended_pdb_1, extended_pdb_2)

        # 3. Assign the protonation states:
        #extended_pdb_3 = os.path.join(self.saving_dir, "extended_pdb3.pdb")
        #self.rename_residue_side_chains(extended_pdb_2, extended_pdb_3, self.dict_mapping)

        # 4. Rename bk atoms:
        #extended_pdb_4 = os.path.join(self.saving_dir, "extended_pdb4.pdb")
        #self.rename_bk_atoms(extended_pdb_3, extended_pdb_4)

        # 5. Rename atom names to Amber format:
        #extended_pdb_5 = os.path.join(self.saving_dir, "extended_pdb5.pdb")
        #self.rename_atomname2amber(extended_pdb_4, extended_pdb_5)

        #if dict_update_protonation_residue is not None:
        #    # 6. Fix protonation states:
        #    extended_pdb_6 = os.path.join(self.saving_dir, "extended_pdb6.pdb")
        #    self.fix_protonation(extended_pdb_5, dict_update_protonation_residue, extended_pdb_6)

            # 7. Keep protein residues and selected ligands
        #    final_output_file = os.path.join(self.saving_dir, pdb_output_name)
        #    self.keep_protein_and_selected_ligands(
        #        extended_pdb_6, final_output_file, self.all_ligands
        #    )

        #    # Delete intermediate files
        #    file_list = [extended_pdb_1, extended_pdb_2, extended_pdb_3, extended_pdb_4, extended_pdb_5, extended_pdb_6]
        #    self.delete_intermediate_files(file_list)
        #    print(f'Conversion is done!')
        #else:
            # 6. Keep protein residues and selected ligands
        final_output_file = os.path.join(self.saving_dir, pdb_output_name)
        self.keep_protein_and_selected_ligands(
            extended_pdb_2, final_output_file, self.all_ligands
        )


        # Delete intermediate files
        file_list = [extended_pdb_1, extended_pdb_2]
        self.delete_intermediate_files(file_list)
        print(f'Conversion is done!')













    def extract_ligands_from_pdb(self, pdb_file_path):
        # Implement the logic to extract ligands from the PDB file and return a list of ligand names
        # For example:
        ligand_list = []
        with open(pdb_file_path, "r") as pdb_file:
            for line in pdb_file:
                ligand_name = line[17:21].strip()
                # print(ligand_name)
                ligand_list.append(ligand_name)
        return set(ligand_list)



    def convert_to_extended_pdb(self, mcce_pdb_path, extended_pdb_path):
        with open(mcce_pdb_path, 'r') as mcce_pdb, open(extended_pdb_path, 'w') as extended_pdb:
            for mcce_line in mcce_pdb:
                extended_line = self.__construct_extended_pdb_line(mcce_line)
                # print(extended_line)
                extended_pdb.write(extended_line)


    @staticmethod
    def __construct_record_name_string(atom_record):
        return f'{atom_record:<6s}'

    @staticmethod
    def __construct_atom_serial_number_string(atom_index):
        # return f'{atom_index:>5f}'
        return f'{str(atom_index):>5s}'

    @staticmethod
    def __construct_atomtype_string(my_atomtype):
        # print(my_atomtype)
        # 13-16 columns (4 columns)
        if len(my_atomtype) > 4:
            raise ValueError(f"atom type can't be me more than 5 characters")
        elif len(my_atomtype) == 4:
            return f'{my_atomtype:<4s}'

        elif len(my_atomtype) == 3:
            return f'{my_atomtype:>4s}'

        elif len(my_atomtype) == 2:
            return f'{my_atomtype:^4s}'

        else:
            return f'{my_atomtype:^4s}'

    @staticmethod
    def __construct_altloc_string():
        return f"{' '}"

    @staticmethod
    def __construct_residue_name_string(residue_name):
        return f'{residue_name:<3s}'

    @staticmethod
    def __construct_chain_id_string(chain_id):
        return f'{chain_id:<1s}'

    @staticmethod
    def __construct_residue_seq_string(residue_index):
        return f'{residue_index:>4s}'

    @staticmethod
    def __construct_icode_string():
        return f'{" "}'

    @staticmethod
    def __construct_xyz_string(my_numb):
        my_numb = float(my_numb)
        if np.abs(my_numb) > 0:
            n_order = np.log10(np.abs(my_numb))
            if n_order > 0:
                n_dig = int(np.floor(n_order)) + 1
            else:
                n_dig = 1
            n_dec = 8 - 2 - n_dig
            form_str = f'>-8.{n_dec:d}f'
            return f'{my_numb:{form_str}}'
        else:
            return f'{my_numb:>8.3f}'

    @staticmethod
    def __construct_spaces_string():
        """
        put spaces un the place of occupancy and temperature factor.
        Returns
        -------

        """
        return f'{" ":21s}'

    @staticmethod
    def __construct_element_symbol_string(element_name):
        return f'{element_name:<2s}'

    @staticmethod
    def __construct_charge_string(charge):
        return f'{charge:5.2f}'

    @staticmethod
    def parse_pdb_line_string(pdb_line):
        record_atom = pdb_line[0:6].strip()
        atom_index = pdb_line[6:12].strip()
        atom_type = pdb_line[12:17].strip()
        residue_name = pdb_line[17:21].strip()
        # chain_name = pdb_line[22:23]
        residue_index = pdb_line[21:26].strip()
        conf_number = pdb_line[27:30].strip()
        x_coord = pdb_line[30:38].strip()
        y_coord = pdb_line[38:47].strip()
        z_coord = pdb_line[47:55].strip()
        if pdb_line[12:16].strip()=='MG':
            element_symbol = 'Mg'
        else:
            element_symbol = pdb_line[12:16].strip()[0]

        return record_atom, atom_index, atom_type, residue_name, conf_number, \
            residue_index, x_coord, y_coord, z_coord, element_symbol


    def __construct_extended_pdb_line(self, mcce_line):
        record_atom, atom_index, atom_type, residue_name, conf_number, \
            residue_index, x_coord, y_coord, z_coord, element_symbol = self.parse_pdb_line_string(
            mcce_line)
        if residue_name not in ['CLA', 'CLB', 'CHL', 'CT1', 'CT2', 'CT3', 'CT4']:
            extended_line = self.__construct_record_name_string(record_atom) + \
                            self.__construct_atom_serial_number_string(atom_index) + \
                            " " * 1 + \
                            self.__construct_atomtype_string(atom_type) + \
                            self.__construct_altloc_string() + \
                            self.__construct_residue_name_string(residue_name) + \
                            " " * 1 + \
                            self.__construct_residue_seq_string(residue_index) + \
                            self.__construct_icode_string() + \
                            " " * 3 + \
                            self.__construct_xyz_string(x_coord) + \
                            self.__construct_xyz_string(y_coord) + \
                            " " * 1 + \
                            self.__construct_xyz_string(z_coord) + \
                            ' ' * 21 + \
                            f'{self.__construct_element_symbol_string(element_symbol)}' + ' ' * 2 + \
                            f'{mcce_line[68:74]}' + ' ' * 5 + \
                            f'{mcce_line[80:82]}' + ' ' + f'{conf_number}' + '\n'
        else:
            extended_line_0 = self.__construct_record_name_string(record_atom) + \
                              self.__construct_atom_serial_number_string(atom_index) + \
                              " " * 1 + \
                              self.__construct_atomtype_string(atom_type) + \
                              self.__construct_altloc_string() + \
                              self.__construct_residue_name_string(residue_name) + \
                              " " * 1 + \
                              self.__construct_residue_seq_string(residue_index) + \
                              self.__construct_icode_string() + \
                              " " * 3 + \
                              self.__construct_xyz_string(x_coord) + \
                              self.__construct_xyz_string(y_coord) + \
                              " " * 1 + \
                              self.__construct_xyz_string(z_coord)

            if element_symbol=='Mg':
                extended_line = extended_line_0 + ' '* 21 + \
                                f'{self.__construct_element_symbol_string(element_symbol)}' + ' '*3 + \
                                'None' + ' '*6 + \
                                f'{mcce_line[80:82]}' + ' ' + f'{conf_number}' + '\n'
            else:
                extended_line = extended_line_0 + ' '* 21 + \
                                f'{self.__construct_element_symbol_string(element_symbol)}' + ' '*3 + \
                                'None' + ' '*6 + \
                                f'{mcce_line[80:82]}' + ' ' + f'{conf_number}' + '\n'
        return extended_line

    def __construct_ligand_extended(self, record_atom, atom_index, atom_type, residue_name, residue_index,
                                    x_coord, y_coord, z_coord, element_symbol, element_charge,
                                    element_protonation_code):
        extended_line = (f'{self.__construct_record_name_string(record_atom)}'
                         f'{self.__construct_atom_serial_number_string(atom_index)}'
                         f'{self.__construct_atomtype_string(atom_type)}'
                         f'{self.__construct_altloc_string()}'
                         f'{self.__construct_residue_name_string(residue_name)}'
                         f'{self.__construct_residue_seq_string(residue_index)}'
                         f'{self.__construct_icode_string()}'
                         f'{self.__construct_xyz_string(x_coord)}{self.__construct_xyz_string(y_coord)} {self.__construct_xyz_string(z_coord)}'
                         f'{self.__construct_spaces_string()}'
                         f'{self.__construct_element_symbol_string(element_symbol)}'
                         f'{self.__construct_charge_string(float(element_charge))}'
                         f'{element_protonation_code}')
        return extended_line

    def rename_capping_residues(self, extended_pdb_path, complex_pdb_path):
        """
        renaming the capping residues that mcce introduce to the terminal residues and the pigment tails.
        """
        capping_residues = ['NTR', 'CTR', 'NTG','CT1','CT2','CT3','CT4']  # List of capping residue names

        with open(extended_pdb_path, 'r') as extended_pdb, open(complex_pdb_path, 'w') as complex_pdb:
            for extended_line in extended_pdb:
                residue_name = extended_line[17:20].strip()

                if residue_name in capping_residues:
                    residue_index = extended_line[21:26].strip()

                    with open(extended_pdb_path, 'r') as pdb_file:
                        for pdb_line in pdb_file:
                            if pdb_line[21:26].strip() == residue_index and pdb_line[17:20].strip() != residue_name:
                                actual_resname = pdb_line[17:20].strip()
                                break

                    complex_line = extended_line.replace(residue_name, actual_resname)
                    complex_pdb.write(complex_line)
                else:
                    complex_pdb.write(extended_line)


    def rename_residue_side_chains(self, pdb_file_path, output_file_path, mapping_dict):
        with open(pdb_file_path, 'r') as pdb_file, open(output_file_path, 'w') as output_file:
            for line in pdb_file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    residue_name = line[17:20].strip()
                    residue_index = line[22:26].strip()
                    key = line[84:88].strip()

                    if residue_name in mapping_dict and key in mapping_dict[residue_name]:
                        new_residue_name = mapping_dict[residue_name][key]
                        line = line[:17] + new_residue_name + line[20:]

                    output_file.write(line)
                else:
                    output_file.write(line)


    @staticmethod
    def filter_dict(dictionary):
        filtered_dict = {}
        for key, values in dictionary.items():
            if values not in filtered_dict.values():
                filtered_dict[key] = values
        return filtered_dict

    @staticmethod
    def __get_residue_info(input_pdb, output_pdb):
        dict_residue = {}
        dict_residue_info = {}

        with open(input_pdb, 'r') as pdb_file, open(output_pdb, 'w') as output_file:
            for line in pdb_file:
                # print(line)
                if line.startswith('ATOM'):
                    residue_id = line[21:26].strip()
                    residue_name = line[17:20].strip()
                    atom_code = line[84:88].strip()

                    if atom_code != 'BK':
                        residue_code = f'{residue_id}_{residue_name}_{atom_code}'
                        if residue_id not in dict_residue:
                            dict_residue[residue_id] = set([residue_code])
                        else:
                            dict_residue[residue_id].add(residue_code)
        return dict_residue

    def rename_bk_atoms(self, pdb_file_path, output_file_path):
        dict_residue_info_mapping = self.__get_residue_info(input_pdb=pdb_file_path, output_pdb=output_file_path)
        # print(dict_residue_info_mapping)
        with open(pdb_file_path, 'r') as pdb_file, open(output_file_path, 'w') as output_file:
            for line in pdb_file:
                if line.startswith('ATOM'):
                    residue_id = line[21:26].strip()
                    residue_name = line[17:20].strip()
                    atom_code = line[84:88].strip()
                    residue_code = f'{residue_name}_{atom_code}'
                    if atom_code == 'BK':
                        if residue_id in dict_residue_info_mapping:
                            new_residue_name = None
                            for code in dict_residue_info_mapping[residue_id]:
                                if code != residue_code:
                                    new_residue_name = list(dict_residue_info_mapping[residue_id])[0].split("_")[1]
                                    break

                            if new_residue_name:
                                line = line.replace(residue_name, new_residue_name)
                    output_file.write(line)
                else:
                    output_file.write(line)


    def remove_ligands(self, pdb_file_path, output_file_path, ligands_to_remove):
        with open(pdb_file_path, 'r') as pdb_file, open(output_file_path, 'w') as output_file:
            for line in pdb_file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    residue_name = line[17:20].strip()
                    if residue_name not in ligands_to_remove:
                        output_file.write(line)
                else:
                    output_file.write(line)


    def rename_atomname2amber(self, pdb_file_path, output_file_path):
        with open(pdb_file_path, "r") as pdb_file, open(
            output_file_path, "w"
        ) as output_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    if (
                        residue_name in amber_atomname_conversion_dict
                        and atom_name in amber_atomname_conversion_dict[residue_name]
                    ):
                        new_atom_name = amber_atomname_conversion_dict[residue_name][
                            atom_name
                        ]
                        line = f"{line[:12]}{self.__construct_atomtype_string(new_atom_name)}{line[16:]}"

                output_file.write(line)


    def construct_altloc_string(self):
        return ''

    def keep_protein_and_selected_ligands(self, pdb_file_path, output_file_path, ligands_to_keep):
        ligands_to_keep = set(ligands_to_keep)

        with open(pdb_file_path, "r") as pdb_file, open(
            output_file_path, "w"
        ) as output_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    residue_name = line[17:20].strip()
                    if residue_name in ligands_to_keep and line.startswith("ATOM"):
                        line = "HETATM" + line[6:]  # Replace "ATOM" with "HETATM"
                    output_file.write(line)
                else:
                    output_file.write(line)



    # def identify_terminal_residues(self, pdb_file_path):
    #     terminal_residues = set()
    #     current_chain = None
    #     with open(pdb_file_path, "r") as pdb_file:
    #         for line in pdb_file:
    #             if line.startswith("ATOM") or line.startswith("HETATM"):
    #                 residue_name = line[17:20].strip()
    #                 residue_index = line[21:26].strip()
    #                 chain_name = line[21]
    #                 if current_chain is None:
    #                     current_chain = chain_name
    #                 elif chain_name != current_chain:
    #                     terminal_residues.add((current_chain, residue_index))
    #                     current_chain = chain_name
    #     return terminal_residues
    # #

    def fix_protonation(self, pdb_path, dict_update_protonation_residue,path):
        """
        Take a PDB file as input and fix protonation issues in the protein and ligand residues based on the provided
        dictionary. It can be used to rename any protein residue or ligand residue that is not defined in GROMACS or
        violates some of GROMACS rules. For example, GROMACS can't take a neutral residue like ASH as a terminal residue,
        so the user can change this residue to be in the deprotonated form 'ASP' to avoid GROMACS termination.

        Parameters
        ----------
        1. pdb_path: str
                     Path of the PDB input.

        2. dict_update_protonation_residue: dict
                                            Keys are the PDB pattern that the user wants
                                            to change (it must start with the chain name
                                            until the residue index), and values are the
                                            new residue names.

        Returns
        -------
        None
        """

        output_pdb_lines = []
        ligands_kept = []
        ligands_removed = []

        with open(pdb_path, "r") as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain_id = line[21]
                    res_seq = line[22:26].strip()
                    key = f"{chain_id}{res_seq}"
                    if key in dict_update_protonation_residue:
                        new_res_name = dict_update_protonation_residue[key]
                        line = line[:17] + new_res_name + line[20:]
                        ligands_kept.append(new_res_name)
                    else:
                        ligands_removed.append(line.split()[2])
                output_pdb_lines.append(line)

        with open(f"{path}fixed_protonation.pdb", "w") as output_pdb:
            output_pdb.writelines(output_pdb_lines)






