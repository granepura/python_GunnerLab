#!/usr/bin/env python
# coding: utf-8

# In[22]:


import sys
import os
import math
import pandas as pd
import numpy as np

conf = 'CHL0AA0406_001.opp'

fort38 = pd.read_csv('fort.38', delimiter=r"\s+")
opp = pd.read_csv(conf, delimiter=r"\s+", names = ['Num', 'Conf', 'AvgPW', 'vdw', 'CorrBCs', 'Delphi', '*'] )
head3 = pd.read_csv('head3.lst', delimiter=r"\s+")
head3.columns = ['Conf', 'FL', 'occ', 'crg', 'Em0', 'pKa0', 'ne', 'nH', 'vdw0', 'vdw1', 'tors', 'epol', 'dsolv', 'extra', 'history','t'] 

print(fort38)
#print(opp)
#print(head3)

for k in head3.itertuples():
    if k[1] == conf[0:14]:
        vdw0 = k[9]
        vdw1 = k[10]
        tors = k[11]
        epol = k[12]
        dsolv = k[13]
        extra = k[14]

Chl = pd.DataFrame([[conf[0:14],vdw0,vdw1,tors,epol,dsolv,extra]])
Chl['Sum'] = vdw0 + vdw1 + tors + epol + dsolv + extra
Chl.columns = ['iConf','vdw0', 'vdw1', 'tors', 'epol', 'dsolv', 'extra', 'Sum']
print(Chl)


# In[45]:


import sys
import os
import pandas as pd
import numpy as np

conf = 'CHL0AA0406_005.opp'

fort38 = pd.read_csv('fort.38', delimiter=r"\s+")
opp = pd.read_csv(conf, delimiter=r"\s+", names = ['Num', 'Conf', 'AvgPW', 'vdw', 'CorrBCs', 'Delphi', '*'] )
head3 = pd.read_csv('head3.lst', delimiter=r"\s+")
head3.columns = ['Conf', 'FL', 'occ', 'crg', 'Em0', 'pKa0', 'ne', 'nH', 'vdw0', 'vdw1', 'tors', 'epol', 'dsolv', 'extra', 'history','t'] 

for k in head3.itertuples():
    if k[1] == conf[0:14]:
        vdw0 = k[9]
        vdw1 = k[10]
        tors = k[11]
        epol = k[12]
        dsolv = k[13]
        extra = k[14]

Chl = pd.DataFrame([[conf[0:14],vdw0,vdw1,tors,epol,dsolv,extra]])
Chl['Sum'] = vdw0 + vdw1 + tors + epol + dsolv + extra
Chl.columns = ['iConf','vdw0', 'vdw1', 'tors', 'epol', 'dsolv', 'extra', 'Sum']
print('This is the Chl Conformer self energies (head3.lst)')
print(Chl)
print('')
        
        
list_1 = []
for i in fort38.itertuples():
    for j in opp.itertuples():
        x = [ "%05d" % (i[0]++1), i[1], j[2], i[2],"{:.3f}".format(j[3]), "{:.3f}".format(j[4]) ]
        
        if i[1] == j[2]: 
            if x not in list_1:
                list_1.append(x)

df = pd.DataFrame(list_1)
df.columns = ['iConf', 'Conf_fort.38','Conf_opp', 'Keq_fort.38', 'AvgPW', 'vdw']
df['RunSum_0vdw'] = ( df['Keq_fort.38'].astype(float) *
                    ( df['AvgPW'].astype(float) + 0*df['vdw'].astype(float) ) 
                    ).cumsum()
df['head3_0vdw0_0vdw1'] = 0*vdw0 + 0*vdw1 + 0.125*tors + epol + dsolv + extra
df['RunSum+head3_0allvdw'] = df['RunSum_0vdw'] + df['head3_0vdw0_0vdw1']
print('This is the Chl Conformer interaction energies (opp)')
print(df)
print('')

Conformer_sum = ( df['Keq_fort.38'].astype(float)* 
                ( df['AvgPW'].astype(float) + 0*df['vdw'].astype(float) )
                ).sum() + 0*vdw0  + 0*vdw1  + 0.125*tors  + epol  + dsolv  + extra
print('Conformer_sum =', "{:.3f}".format(Conformer_sum) )

mfeplus_sum = ( df['Keq_fort.38'].astype(float)*
              ( df['AvgPW'].astype(float) + df['vdw'].astype(float) ) 
              ).sum()
print('mfeplus_sum=', "{:.3f}".format(mfeplus_sum) )

#list_2 = []
#for k in head3.itertuples():
#    for l in df.itertuples():
#        z = [ "%05d" % (k[0]), k[1], l[4], l[5], vdw0, vdw1, tors, epol, dsolv, extra ]
#        
#        if k[1] == l[2]:
#            if z not in list_2:
#                list_2.append(z)
                    
#df2 = pd.DataFrame(list_2)
#df2.columns = ['iConf', 'Conf', 'AvgPW', 'vdw', 'vdw0', 'vdw1', 'tors', 'epol', 'dsolv', 'extra']

#mfeplus_sum = df2['AvgPW'].astype(float).sum() + df2['vdw'].astype(float).sum()
#Conformer_sum = df2['AvgPW'].astype(float).sum() + 0*df2['vdw'].astype(float).sum() + 0*vdw0  + 0*vdw1  + 0.125*tors  + epol  + dsolv  + extra
#print('mfeplus_sum=', mfeplus_sum)
#print('Conformer_sum =', Conformer_sum)

#print(mfe)


# In[38]:


# Total Conformer Interaction Energy - 0vdw
TCIE = pd.DataFrame([
    ['CHL0AA0406_001', 9.143, ],
    ['CHL0AA0406_002', 7.096, ],
    ['CHL0AA0406_003', 6.120, ],
    ['CHL0AA0406_004', 8.967, ],
    ['CHL0AA0406_005', 6.802, ]
    ])
TCIE.columns = ['Conf', 'Gibbs'] 

TCIE_norm = pd.DataFrame([
    ['CHL0AA0406_001', 9.143, ],
    ['CHL0AA0406_002', 7.096, ],
    ['CHL0AA0406_003', 6.120, ],
    ['CHL0AA0406_004', 8.967, ],
    ['CHL0AA0406_005', 6.802, ]
    ])
TCIE_norm.columns = ['Conf', 'Gibbs']


TCIE_norm['BE'] = np.power(10, -TCIE_norm['Gibbs']/1.36)
TCIE_norm['Partition_BE'] = TCIE_norm['BE'].sum()
TCIE_norm['BoltzAvgDist'] = TCIE_norm['BE']/TCIE_norm['Partition_BE']

print(TCIE_norm)


# In[20]:


# Total Conformer Interaction Energy - 0.01vdw
TCIE = pd.DataFrame([
    ['CHL0AA0406_001', 1529.299, ],
    ['CHL0AA0406_002', 1415.476, ],
    ['CHL0AA0406_003', 1432.344, ],
    ['CHL0AA0406_004', 1485.875, ],
    ['CHL0AA0406_005', 1509.630, ]
    ])
TCIE.columns = ['Conf', 'Gibbs'] 

TCIE_norm = pd.DataFrame([
    ['CHL0AA0406_001', 1529.299-1415.476, ],
    ['CHL0AA0406_002', 1415.476-1415.476, ],
    ['CHL0AA0406_003', 1432.344-1415.476, ],
    ['CHL0AA0406_004', 1485.875-1415.476, ],
    ['CHL0AA0406_005', 1509.630-1415.476, ]
    ])
TCIE_norm.columns = ['Conf', 'Gibbs']


TCIE_norm['BE'] = np.power(10, -TCIE_norm['Gibbs']/1.36)
TCIE_norm['Partition_BE'] = TCIE_norm['BE'].sum()
TCIE_norm['BoltzAvgDist'] = TCIE_norm['BE']/TCIE_norm['Partition_BE']

print(TCIE_norm)

