#!/usr/bin/env python
from sys import argv
# ftpl file should be like res.ftpl
step2_file = argv[1]
def change_step2_crg(step2_file):
    # give the param_path here 
    param_path = "/home/granepura/Stable-MCCE/param_PARSE/"
    lines = open(step2_file, 'r')
    dict_ftpl_crg = {}
    for line in lines:
        # check the line and open the residues
        res_step2 = line[17:20]
        conf_step2 = line[80:82]
        crg_step2 = line[68:74]
        atom_step2 = line[12:16].strip()
        step2_ident = res_step2 + conf_step2+"_"+ atom_step2
        # only open ftpl file if not already open
        if step2_ident not in dict_ftpl_crg:
            with open(param_path+res_step2.lower()+".ftpl") as f: # ftpl file open: file format: arg.ftpl
                for l in f.readlines():
                 if l[:6] == "CHARGE":
                    line_split = l.strip().split(":")
                    charge = line_split[1].split("#")[0].strip() # if # will split else give the value
                    conf_infor = line_split[0].split(",")
                    atom_ftpl = conf_infor[2].replace('"', '').strip()
                    conf_atm = conf_infor[1].strip() +"_"+ atom_ftpl
                    dict_ftpl_crg[conf_atm] = charge

        crg_pos = f"{dict_ftpl_crg.get(step2_ident):>6}"
        #out.write(line[:68] + crg_pos + line[74:].rstrip())
        print(line[:68] + crg_pos + line[74:].rstrip())
    


if __name__ == "__main__":
    change_step2_crg(step2_file)




            
