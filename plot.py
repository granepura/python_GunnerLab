# This script plots user inputted residue arguments from sum_crg.out against its pH
# To run code use command
# python plot.py sum_crg.out <residue_001> <residue_002> <residue_003>

import sys # Get output outside program

#print(sys.argv[1])
#python plot.py sum_crg.out 
f = open(sys.argv[1], 'r') # Default is read ('r'), 1 is first file run after filename
pH = []
allaminoacids={}

for i in f:
	#print(i)
	t = i.split() 					# Returns an array of all elements in line
	aa=[]
	if len(t)>2:                                     # Ensures empty lines are not outputted  
		if t[0]=='pH':   
			for j in range(1, len(t)):
				pH.append(float(t[j])) 	
		else:
			for j in range(1, len(t)):
				aa.append(float(t[j]))
			allaminoacids[t[0]]=aa
	#print(t)	

print(pH)
# print(type(pH))  # prints type of file
for i in allaminoacids.keys():
	print(i, allaminoacids[i])
f.close()

import matplotlib.pyplot as plt
#for i in allaminoacids.keys():
#	plt.plot(pH, allaminoacids[i])

#for i in allaminoacids.keys():
#plt.plot(pH, allaminoacids[sys.argv[2]])
#plt.xlabel("pH")
#plt.ylabel("Charge")
#plt.title(sys.argv[2])
#plt.show()

for i in range(2, len(sys.argv)):
	plt.plot(pH, allaminoacids[sys.argv[i]], label=sys.argv[i])

plt.legend()
plt.xlabel("pH")
plt.ylabel("Charge")
plt.title("sum_charge.out")
plt.show()	
