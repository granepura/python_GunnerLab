# Here you need:
# mccce output file:  ms_state file(for ex: pH7eH0ms.txt), head3.lst, step2_out.pdb

# script: ms_analysis.py

import ms_analysis as msa
import numpy as np
import os

# put your directory here, remember to also update head3.lst path in ms_analysis.py
# path to the MCCE directory data
# path to ms_out file data
# mcce_dir = '/Users/mohamedelrefaiy/Library/CloudStorage/Box-Box/Reseach/projects/2023/ISIA/MCCE/Aug_27_2023/trimer'
mcce_dir = "."
mc = msa.MSout(f"{mcce_dir}/ms_out/pH7eH0ms.txt")

ms_orig_lst = [[ms.E, ms.count, ms.state] for ms in list((mc.microstates.values()))]
ms_orig_lst = sorted(ms_orig_lst, key=lambda x: x[0])



def writeMS2PDB_single(ms_selection, ms_index, step2_path, output_folder):
    """
    ms_list is the conformer microstate information not charge state ms.
    ms_start gives the flexibilty from which state you want to write pdb file.
    skip_step_ms = 0 means write out all the ms state. You can skip any number.
    You must have the step2_out.pdb file to run this code. Make sure give the path for
    step2_out.pdb for step2_path. output_folder where you want to write pdb file.
    """

    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder)

    all_conf_list = []
    for conf in msa.conformers:
        if conf.iconf in ms_selection[2] or conf.iconf in mc.fixed_iconfs:
            all_conf_list.append(conf.confid)
    pdb = open(step2_path).readlines()
    file_name = f"{output_folder}/ms_pdb_{ms_index}.pdb"
    with open(file_name, 'w') as output_pdb:
        for line in pdb:
            if len(line) < 82: continue
            if line[26] == ' ':
                iCode = '_'
            else:
                iCode = line[26]
            confID = line[17:20] + line[80:82] + line[21:26] + iCode + line[27:30]

            if confID in all_conf_list or confID[3:5] == 'BK':
                output_pdb.write(line)
    print(f"done")
    return


n_counts = 0.
ms_count_values = []
for ms in ms_orig_lst:
    n_counts += ms[1]
    ms_count_values.append(ms[1])

full_count_list = list(np.arange(n_counts))

ms_cum_sum = np.cumsum(ms_count_values)

count_selection_list = np.arange(100, n_counts - 100, n_counts / 100)


for count in count_selection_list:
    ms_index = np.where((ms_cum_sum - count) > 0)[0][0]
    conf_ms_write = ms_orig_lst[ms_index]
    step2_path = f"{mcce_dir}/step2_out.pdb"
    pdb_out_folder = f"{mcce_dir}/pdb_output_mc/"
    # write out the pdb in the folder low_high_avg_pdb
    writeMS2PDB_single(ms_selection=conf_ms_write, ms_index=ms_index,
                       step2_path=step2_path, output_folder=pdb_out_folder)

print('Calculation done!')
