import os
import pandas as pd

def extract_residue_data(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    residue_data = {}
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            residue_id_part1 = line[16:20].strip()
            residue_id_part2 = line[20:26].strip()
            characters_94_97 = line[93:97].strip()

            # Only add residues where characters 94-97 are not '000'
            if characters_94_97 != '000':
                residue_label = f"{residue_id_part1}_{residue_id_part2}"
                if residue_label not in residue_data:
                    residue_data[residue_label] = characters_94_97

    return residue_data

def main():
    pdb_directory = 'parsed_pdb_output_mc'
    matrix = {}

    pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith('.pdb')]

    # Loop through all PDB files in the directory
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_directory, pdb_file)

        # Extract residue data for the current PDB file
        residue_data = extract_residue_data(pdb_path)

        # Add residue data to the matrix
        matrix[pdb_file] = residue_data

    # Convert the matrix to a pandas DataFrame
    df = pd.DataFrame.from_dict(matrix, orient='index')

    # Remove columns (residues) where all elements are the same
    df_free = df.loc[:, (df != df.iloc[0]).any()]

    # Save the filtered DataFrame to a new CSV file
    df_free.to_csv('free_residue_matrix.csv')

if __name__ == '__main__':
    main()

