import sys

# This script creates PDB files for every conformer of every residue in MCCE's step2_out.pdb file
# To run the script, first create a pdb file containing all the confomers for residue(s) in MCCE step2_out.pdb format  
# $ python MCCEstep2_conf2pdbs.py RESI_step2_out.pdb 

def create_pdb_files(pdb_data):
    # Split the step2_out PDB data into lines
    # Create dictionary to store atom lines for each conformer
    pdb_lines = pdb_data.strip().split('\n')
    atom_lines_dict = {}

    # Group MCCE conformers for atom lines based on characters 21-29
    for line in pdb_lines:
        chain = line[21]
        res   = line[17:20]
        conf_chain  = line[21:30].strip()
        conf = conf_chain[1:] 
        state = line[80:82]
        name = f"{chain}-{res}{conf}_{state}"
        if name not in atom_lines_dict:
            atom_lines_dict[name] = []
        atom_lines_dict[name].append(line)

    # Create PDB files for each MCCE conformer
    for key, lines in atom_lines_dict.items():
        pdb_file_name = f"{key}.pdb"
        with open(pdb_file_name, 'w') as pdb_file:
            pdb_file.write('\n'.join(lines))

        print(f"Created PDB file: {pdb_file_name}")

if __name__ == "__main__":
    # Check if the step2_out PDB data file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py pdb_data_file")
        sys.exit(1)

    pdb_file_path = sys.argv[1]

    # Read step2_out PDB data from the file
    try:
        with open(pdb_file_path, 'r') as pdb_file:
            pdb_data = pdb_file.read()
            create_pdb_files(pdb_data)
    except FileNotFoundError:
        print(f"Error: File not found: {pdb_file_path}")
        sys.exit(1)


