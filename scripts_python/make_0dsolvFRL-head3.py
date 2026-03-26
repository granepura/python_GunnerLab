#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

input_file  = "0dsolvFRL_head3.lst"
output_file = "0dsolvFRL-PSII_head3.lst"

# Duplicate the first conformer line of head3.lst
with open(input_file) as file, open("duplicated_file.txt", "w") as new_file:
    first_line = file.readline()
    second_line = file.readline()
    new_file.write(first_line)
    new_file.write(second_line)
    new_file.write(second_line)
    for line in file:
        new_file.write(line)        
input_file = "duplicated_file.txt"

# Open Dataframe with dupilicated head3.lst and remove the 1st line (header)
df = pd.read_csv(input_file, skiprows=1, delim_whitespace=True)

# Replace dupilcated row1 with the header
df.columns = ["iConf", "CONFORMER", "FL", "occ", "crg", "Em0", "pKa0", "ne", "nH", "vdw0", "vdw1", "tors", "epol", "dsolv", "extra", "history", " "]

# Replace value in 'dsolv' column with 0.000 if 'CONFORMERS' column contains 'AvB'
df.loc[df['CONFORMER'].str.contains("FRL"), 'dsolv'] = 0.000

df.loc[df['CONFORMER'].str.contains("FRL0D"), 'extra'] = 2.675
df.loc[df['CONFORMER'].str.contains("FRL0F"), 'extra'] = 1.303
df['iConf'] = df['iConf'].astype(str).str.zfill(5)


# Convert dataframe to string
df_string = df.to_string(index=False)
print(df_string)

# Write dataframe string to .lst file
with open(output_file, "w") as file:
    file.write(df_string)


# In[ ]:




