#!/bin/bash

# Use pymol to find all HOH within 5 angstroms of CHLs
# PyMOL > select resn HOH & (all within 5 of resn CLA)
# PyMOL > select resn HOH & (all within 5 of resn CL7)
# PyMOL > select resn HOH & (all within 5 of resn F6C)
# PyMOL --> File > Export Molecule > Selection as sele

file="HOHs_center.pdb"
rm -rf $(awk '{print substr($4,1,3) "-" substr($5,1,1) "-" substr($6,1,3)}' $file)
mkdir $(awk '{print substr($4,1,3) "-" substr($5,1,1) "-" substr($6,1,3)}' $file)


for dir in */; do
	echo "$dir"
	cp makeHOHconf_GR.py $dir/  
        grep "$(echo "$dir" | sed 's,\-, ,g' | sed 's,\/,,g')" $file > $dir/wat.pdb  
	python $dir/makeHOHconf_GR.py $dir/wat.pdb > $dir/HOH_coor.txt 
	mv HOH_confs.pdb $dir/
done

cat HOH*/HOH_confs.pdb > "${file%.*}_step2_out.${file##*.}"
