#!/usr/bin/env python
"""
Step 3 computes energy look up table.
The input is step2_out.pdb.
The output is energies/*.opp, energies/*.oppl and head3.lst

Command line options:
-d number: Dielectric constant (default 4)
-s PBSolver: PBE solver name (default ngpb, choices=["delphi", "ngpb", "zap", "apbs", "template"])
-p number: Run with number of threads (default 1)
-c start end: Run conformer from start to end (default 0 to 99999)
-t path: Temporary folder path (default /tmp)
-ftpl: ftpl folder
-salt: Salt concentration in moles/L
-vdw_relax: Relax vdw R parameter by +- specified value
--fly: On-the-fly rxn0 calculation
--skip_pb: Run vdw and torsion calculation only
--debug: Print debug information and keep pbe solver tmp
--refresh: Recreate *.opp and head3.lst from step2_out.pdb and *.oppl files without doing calculation
-l file: Load above options from a file
-load_runprm: Load additional run.prm file, overwrite default values

Usage examples:

1. Run step 3 with 6 threads
    step3.py -p 6

"""
import argparse
from collections import defaultdict
import json
import logging
from multiprocessing import Pool, current_process, TimeoutError
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from mccesteps import record_runprm 
from mccesteps import export_runprm
from mccesteps import detect_runprm
from mccesteps import restore_runprm
from pbs_interfaces import *

# Set VDW function
if "--old_vdw" in sys.argv:
    from pdbio import *
else:
    from pdbio_gr import *


logger = logging.getLogger("step3.py")
logger.setLevel(logging.DEBUG)


energy_folder = "energies"
PW_CUTOFF = 0.001  # cut off value for pairwise interaction to report
PROGRESS_LOG = "progress_step3.log"
PBE_TASK_TIMEOUT = 1000  # seconds per conformer; rerun stragglers with -c

global run_options
global instance_name 


# copied from mcce4.io_utils:
def config_logger(step_num: int, log_level: str = "INFO"):
    """Function to configure a logger for MCCE step modules.
    Configuration for logging to screen and files: run.log, err.log, and
    'step<step_num>.debug' if log_level is 'DEBUG'.

    Args:
      step_num (str):
        Number of the MCCE step to configure for logging to file when log_level = 'DEBUG'.
      log_level (str, "DEBUG")
        If 'DEBUG' a file handler is created for the given step, e.g. 'step1.debug'.

    Returns:
      logging.Logger: The configured logger object.
    """
    choices = list(logging._nameToLevel.keys())
    log_level = log_level.upper()
    if log_level not in choices:
        log_level = "INFO"
        print(f"log_level must be one of {choices}; reset to INFO")

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with INFO level and a more concise formatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(name)s: %(message)s"))
    
    # File handler for 'run.log'
    run_fh = logging.FileHandler("run.log", encoding="utf-8")
    run_fh.setLevel(logging.INFO)
    run_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(name)s, %(funcName)s: %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
                                          )

    # File handler for errors with a more detailed formatter
    err_fh = logging.FileHandler("err.log", encoding="utf-8")
    err_fh.setLevel(logging.ERROR)
    err_format = "[%(asctime)s - %(levelname)s]: %(name)s, %(funcName)s: %(message)s"
    detail_frmtr = logging.Formatter(err_format, datefmt="%Y-%m-%d %H:%M:%S")
    err_fh.setFormatter(detail_frmtr)
    # Add a filter to ensure only ERROR level messages are logged
    # err_fh.addFilter(lambda record: record.levelno == logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(run_fh)
    logger.addHandler(err_fh)

    if log_level == "DEBUG":
        # output everything to file
        fname = f"step{step_num}.debug"
        dbg = logging.FileHandler(fname, encoding="utf-8")
        dbg.setLevel(logging.DEBUG)
        dbg.setFormatter(detail_frmtr)
        logger.addHandler(dbg)

    return logger


class ElePW:
    def __init__(self):
        self.mark = ""
        self.multi = 0.0
        self.single = 0.0
        self.scaled = 0.0
        self.averaged = 0.0
        return


class RunOptions:
    def __init__(self, args):
        self.inputpdb = "step2_out.pdb"  # implicit input pdb file
        self.start = args.c[0]
        self.end = args.c[1]
        self.d = float(args.d)
        self.do = float(args.do)
        self.s = args.s
        self.p = args.p
        self.t = args.t
        
        # change self.t to absolute path
        self.t = Path(self.t).resolve()

        self.ftpl = args.ftpl
        self.salt = args.salt
        self.skip_pb = args.skip_pb
        self.fly = args.fly
        self.debug = args.debug
        self.refresh = args.refresh
        if args.l:  # load options from the specified file
            lines = open(args.l).readlines()
            for line in lines:
                line = line.split("#")[0].strip()
                # Extract an option line and strip off the spaces:
                fields = [x.strip() for x in line.split()]
                if len(fields) > 0:
                    key = fields[0]
                    if key == "-c":
                        self.start = int(fields[1])
                        self.end = int(fields[2])
                    elif key == "-d":
                        self.d = float(fields[1])
                    elif key == "-do":
                        self.do = float(fields[1])
                    elif key == "-s":
                        self.s = fields[1]
                    elif key == "-p":
                        self.p = int(fields[1])
                    elif key == "-t":
                        self.t = fields[1]
                    elif key == "--vdw":
                        self.vdw = True
                    elif key == "--fly":
                        self.fly = True
                    elif key == "--refresh":
                        self.refresh = True
                    elif key == "--debug":
                        self.debug = True
                    elif key == "-ftpl":
                        self.ftpl = fields[1]

        if args.load_runprm:  # load additional run.prm file
            with open(args.load_runprm) as lines:
                for line in lines:
                    entry_str = line.strip().split("#")[0]
                    fields = entry_str.split()
                    if len(fields) > 1:
                        key_str = fields[-1]
                        if key_str[0] == "(" and key_str[-1] == ")":
                            key = key_str.strip("()").strip()
                            value = fields[0]
                            setattr(self, key, value)

        if args.u:
            fields = args.u.split(",")
            for field in fields:
                try:
                    key, value = field.split("=")
                    setattr(self, key, value)
                except ValueError:
                    print(f"Each component format is 'KEY=VALUE'. Unrecognized: {field}.")


        # unconditionally force to run step 3 only in step3.py
        setattr(self, "DO_ENERGY", "t")
        setattr(self, "MCCE_HOME", str(Path(__file__).parent.parent))

        # convert attributes to runprm dict
        runprm = {}
        attributes = vars(self)
        for key, entry in attributes.items():
            if not key.startswith("_"):
                runprm[key] = entry
        record_runprm(runprm, "#STEP3")
        export_runprm(runprm)

    def toJSON(self):
        return json.dumps(self.__dict__, indent=4)


class ExchangeAtom:
    def __init__(self, atom):
        self.x = atom.xyz[0]
        self.y = atom.xyz[1]
        self.z = atom.xyz[2]
        self.r = atom.r_bound  # radius
        self.c = 0.0  # default to boundary defining atom, charge should be set as 0
        self.p = 0.0
        return


class Exchange:
    """
    Holds the data passed to the PB wrapper, together with runoptions.
    """
    # We have to abandon the mop on and off mechanism when modifying the dielectric boundary.
    # In the parallel for loop, we don't want the boundary revising to be dependent on previous step.
    # Each step in the loop should be an addition or deletion from a static starting point.
    # Therefore, we will compose
    # * backbone atoms
    # * index to match multiple atoms to boundary line number
    # * method to compose single side chain condition
    # * method to compose multi side chain condition

    def __init__(self, protein):
        # initilaize backbone:
        self.backbone_xyzrcp = []
        self.backbone_atom = []
        for res in protein.residue:
            if res.conf:
                for atom in res.conf[0].atom:
                    xyzrcp = ExchangeAtom(atom)
                    self.backbone_xyzrcp.append(xyzrcp)
                    self.backbone_atom.append([atom])
                    # the atom is in a list because it is allowed to have multiple
                    # atoms to match the same line in xyzrcp line

        self.float_bnd_xyzrcp = []
        self.float_bnd_atom = []
        self.single_bnd_xyzrcp = []
        self.single_bnd_atom = []
        self.multi_bnd_xyzrcp = []
        self.multi_bnd_atom = []
        return

    def compose_float(self, protein, ir, ic):
        """Compose a floating side chain boundary condition for rxn0 calculation.
        Only atoms in residue[ir], conformer[ic] are in this list
        """
        self.float_bnd_xyzrcp = []
        self.float_bnd_atom = []

        # Add backbone of a residue for floating boundary
        for atom in protein.residue[ir].conf[0].atom:
            xyzrcp = ExchangeAtom(atom)
            xyzrcp.c = 0.0
            if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                xyzrcp.r = atom.float_born_radius
            self.float_bnd_xyzrcp.append(xyzrcp)
            self.float_bnd_atom.append([atom])

        for atom in protein.residue[ir].conf[ic].atom:
            xyzrcp = ExchangeAtom(atom)
            xyzrcp.c = atom.charge
            if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                xyzrcp.r = atom.float_born_radius
            self.float_bnd_xyzrcp.append(xyzrcp)
            self.float_bnd_atom.append([atom])

        return

    def compose_single(self, protein, ir, ic):
        """Compose a single side chain boundary condition.
        The atoms are added in addition to backbone.
        Atoms other than in residue[ir], conformer[ic] are then appended.
        Atoms in residue[ir], conformer[ic] are appended last
        """
        self.single_bnd_xyzrcp = self.backbone_xyzrcp.copy()
        self.single_bnd_atom = self.backbone_atom.copy()

        for ires in range(len(protein.residue)):
            # print(protein.residue[ires].resID)
            if ires == ir:  # this is the residue we want to put desired side chain conf
                for atom in protein.residue[ires].conf[ic].atom:
                    xyzrcp = ExchangeAtom(atom)
                    xyzrcp.c = atom.charge
                    if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                        xyzrcp.r = atom.born_radius
                    self.single_bnd_xyzrcp.append(xyzrcp)
                    self.single_bnd_atom.append([atom])
            else:
                # find the first charged conformer if any, otherwise use the first
                # skip dummy or backbone only residue:
                if len(protein.residue[ires].conf) > 1:
                    i_useconf = 1  # defaul the first conformer
                    for iconf in range(1, len(protein.residue[ires].conf)):
                        # print(protein.residue[ires].conf[iconf].confID, protein.residue[ires].conf[iconf].crg)
                        if abs(protein.residue[ires].conf[iconf].crg) > 0.0001:
                            i_useconf = iconf
                            break
                    # just for boundary
                    for atom in protein.residue[ires].conf[i_useconf].atom:
                        xyzrcp = ExchangeAtom(atom)
                        if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                            xyzrcp.r = atom.born_radius
                        self.single_bnd_xyzrcp.append(xyzrcp)
                        self.single_bnd_atom.append([atom])

        # Error checking, basic
        # for atom_list in self.ibound2atoms:
        #     print(len(atom_list), atom_list[0].atomID)
        #     if len(atom_list) != 1:
        #         print("ERROR")
        #         break
        if len(self.single_bnd_xyzrcp) != len(self.single_bnd_atom):
            logger.critical(
                "%s Single-sidechain boundary record length should be equal: %d, %d"
                % (protein.residue[ir].conf[ic].confID,
                   len(self.single_bnd_xyzrcp),
                   len(self.single_bnd_atom),
                   )
                )
            sys.exit("Single-sidechain boundary record length should be equal.")

        return

    def compose_multi(self, protein, ir, ic):
        """Compose a multi side chain boundary condition.
        The atoms are added in addition to backbone.
        Atoms other than in residue[ir], conformer[ic] are appended next.
        Atoms in residue[ir], conformer[ic] are appended last.
        When appending an atom, this subroutine will check if the same atom
        (xyzrc identical) already exists. If yes, just update icound
        """
        self.multi_bnd_xyzrcp = self.backbone_xyzrcp.copy()
        self.multi_bnd_atom = self.backbone_atom.copy()

        for ires in range(len(protein.residue)):
            # print(protein.residue[ires].resID)
            if ires == ir:
                # this is the residue we want to put desired side chain conf
                for atom in protein.residue[ires].conf[ic].atom:
                    xyzrcp = ExchangeAtom(atom)
                    xyzrcp.c = atom.charge
                    if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                        xyzrcp.r = atom.born_radius
                    self.multi_bnd_xyzrcp.append(xyzrcp)
                    self.multi_bnd_atom.append([atom])

            else:
                # other residues will have all conformers with 0 charge.
                # skip dummy or backbone only residue
                if len(protein.residue[ires].conf) > 1:
                    # this is the xyzrcp record for all side chain atoms of this residue
                    residue_bnd_xyzrcp = []
                    # this points to the atom records of each line in residue_bnd_xyzrpc
                    residue_bnd_atom = []
                    for iconf in range(1, len(protein.residue[ires].conf)):
                        for atom in protein.residue[ires].conf[iconf].atom:
                            xyzrcp = ExchangeAtom(atom)
                            if run_options.s.upper() == "ML":  # overwrite the r_bound by ML number
                                xyzrcp.r = atom.born_radius
                            # test if this atom existed within this residue already
                            found = False
                            for ib in range(len(residue_bnd_xyzrcp)):
                                if (abs(residue_bnd_xyzrcp[ib].x - xyzrcp.x) < 0.001
                                        and abs(residue_bnd_xyzrcp[ib].y - xyzrcp.y) < 0.001
                                        and abs(residue_bnd_xyzrcp[ib].z - xyzrcp.z) < 0.001
                                        and abs(residue_bnd_xyzrcp[ib].r - xyzrcp.r) < 0.001):
                                    # identical atom
                                    residue_bnd_atom[ib].append(atom)
                                    found = True
                                    break

                            if not found:
                                residue_bnd_xyzrcp.append(xyzrcp)
                                residue_bnd_atom.append([atom])

                    # merge this residue to the multi-bnd
                    self.multi_bnd_xyzrcp += residue_bnd_xyzrcp
                    self.multi_bnd_atom += residue_bnd_atom

        # Basic error checking
        if len(self.multi_bnd_xyzrcp) != len(self.multi_bnd_atom):
            logger.critical(
                "%s Multi-sidechain boundary record length should be equal: %d, %d" % (
                    protein.residue[ir].conf[ic].confID,
                    len(self.multi_bnd_xyzrcp),
                    len(self.multi_bnd_atom))
            )
            sys.exit("Multi-sidechain boundary record length should be equal.")

        return

    def write_boundary(self, fname: str, kind: str):
        """This writes out both the xyzrcp and atom index files, for error
        checking and potentially as data exchange with PB wrapper.
        Args:
         - fname: Name for stem of file (preceding the '.xyzrcp' or '.atoms' extension)
         - kind (str): One of ["float", "single", "multi"]
        """
        valid_kinds = ["float", "single", "multi"]
        if kind not in valid_kinds:
            raise ValueError(f"'kind' must be one of {valid_kinds}; given: {kind}.")

        if kind == "float":
            bnd_xyzrcp = self.float_bnd_xyzrcp
            bnd_atom = self.float_bnd_atom
        elif kind == "single":
            bnd_xyzrcp = self.single_bnd_xyzrcp
            bnd_atom = self.single_bnd_atom
        else:
            bnd_xyzrcp = self.multi_bnd_xyzrcp
            bnd_atom = self.multi_bnd_atom

        # write xyzrcp
        xyzrcp_frmt = "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}\n"
        with open(fname + ".xyzrcp", "w") as ofh:
            for xp in bnd_xyzrcp:
                ofh.write(xyzrcp_frmt.format(xp.x, xp.y, xp.z, xp.r, xp.c, xp.p))

        # write index file
        xyzrc_frmt = "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}"
        with open(fname + ".atoms", "w") as ofh:
            for matched in bnd_atom:
                xyzrc = xyzrc_frmt.format(
                    matched[0].xyz[0],
                    matched[0].xyz[1],
                    matched[0].xyz[2],
                    matched[0].r_bound,
                    matched[0].charge,
                )
                ofh.write(f"{xyzrc} {' '.join([atom.atomID for atom in matched])}\n")

        return


def reset_ligated_pw():
    """
    Reset pairwise interaction involving ligated ligands to 0.
    Example: Heme and ligands together define its redox potential. So the pairwise interaction between ligands and heme should be 0.
    """

    # Helper classes and functions
    class Atom:
        def __init__(self):
            self.icount = 0
            self.name = ""
            self.element = ""
            self.resname = ""
            self.chainid = ""
            self.seqnum = 0
            self.icode = ""
            self.xyz = ()
            self.confname = ""
            return

        def loadline(self, line):
            self.icount = int(line[6:11])
            self.name = line[12:16]
            self.resname = line[17:20]
            self.chainid = line[21]
            self.seqnum = int(line[22:26])
            self.icode = line[26]
            self.xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            self.confname = "%3s%2s%9s" % (self.resname, line[80:82], line[21:30])

            if len(self.name.strip()) >= 4 and self.name[0] == " H":
                self.element = " H"
            elif self.name[:2] == " H":
                self.element = " H"
            else:
                self.element = self.name[:2]

            return


    def dvv(v1, v2):
        """Return Euclidean distance between two 3D vectors."""
        v1a = np.asarray(v1)
        v2a = np.asarray(v2)
        return float(np.linalg.norm(v1a - v2a))

    def shortest_d(res1_atoms, res2_atoms):
        dmin = 1.0E10
        for atom1 in res1_atoms:
            if atom1.element == " H":
                continue
            for atom2 in res2_atoms:
                if atom2.element == " H":
                    continue
                dd = dvv(atom1.xyz, atom2.xyz)
                if dmin > dd:
                    dmin = dd
        return dmin

    def match_conf(atom, conformers):
        matched = []
        for conf in conformers:
            if (conf[:3], conf[5], int(conf[6:10]), conf[10]) == atom.resid:
                matched.append(conf)
        return matched

    def set0(fname, conformer):
        if os.path.exists(fname):
            opplines = open(fname).readlines()
            newlines = []
            for line in opplines:
                fields = line.split()
                if len(fields) > 3:
                    if fields[1] == conformer:
                        newline = "%s  +0.000   0.000   0.000   0.000\n" % (line[:21])
                        newlines.append(newline)
                    else:
                        newlines.append(line)
                else:
                    newlines.append(line)
            open(fname, "w").writelines(newlines)

        return

    # Define constants and parameters
    opp_dir = "energies"
    step2_out = "step2_out.pdb"
    BOND_threshold = 2.7
    ligands = ["HIL", "MEL"]
    receptors = ["HEB", "HEC", "HEM", "CLA"]
    
    # Collect ligand conformers from oppe_dir
    # list all opp files in opp_dir and find those belong to ligands and receptors
    ligand_confs = []
    receptor_confs = []
    for fname in os.listdir(opp_dir):
        if fname.endswith(".opp"):
            resname = fname[:3]
            if resname in ligands:
                ligand_confs.append(Path(fname).stem)
            elif resname in receptors:
                receptor_confs.append(Path(fname).stem)

    atoms = []
    lines = open(step2_out).readlines()
    for line in lines:
        if line[:6] == "ATOM  " or line[:6] == "HETATM":
            atom = Atom()
            atom.loadline(line)
            atoms.append(atom)

    for ligand_conf in ligand_confs:
        ligand_conf_atoms = []
        for atom in atoms:
            if ligand_conf == atom.confname:
                ligand_conf_atoms.append(atom)
        for receptor_conf in receptor_confs:
            receptor_conf_atoms = []
            for atom in atoms:
                if receptor_conf == atom.confname:
                    receptor_conf_atoms.append(atom)
            if shortest_d(ligand_conf_atoms, receptor_conf_atoms) < BOND_threshold:
                set0("energies/%s.opp" % ligand_conf, receptor_conf)
                set0("energies/%s.opp" % receptor_conf, ligand_conf)





def calculate_born_radius(protein):
    """
    Calculate Born radius for atoms with Machine Learning
    Background atoms are the native atoms (conf 0 and 1)
    Background atoms born radii are calculated at background settings
    Non-background atoms (conf 1 and above) are using background atoms as reference.
    """
    from ml_pbs import ML_Solver
    from ml_pbs import ATOM_RADII, ATOM_RADIUS_UNKNOWN

    # Revise r_bound of atoms
    for res in protein.residue:
        for conf in res.conf:
            for atom in conf.atom:
                if atom.name.startswith("H") and len(atom.name) == 4:
                    element = " H"
                else:
                    element = atom.name[0:2]
                # Overwrite r_bound with atom radius
                if element in ATOM_RADII:
                    atom.r_bound = ATOM_RADII[element]
                else:
                    atom.r_bound = ATOM_RADIUS_UNKNOWN

    # Efficiently initialize born_radius and float_born_radius for all atoms
    for res in protein.residue:
        for conf in res.conf:
            for atom in conf.atom:
                atom.born_radius = atom.float_born_radius = 0.0

    # Collect background conformers
    logger.info("   Working on native conformers conf[0] and conf[1].")
    background_conformers = []
    for res in protein.residue:
        for conf in res.conf[:2]:
            background_conformers.append(conf)
    # collect atoms from conformers and make a pqr list
    background_atoms = []
    for conf in background_conformers:
        background_atoms.extend(conf.atom)
    pqr_data = []
    for atom in background_atoms:
        pqr_data.append((atom.xyz[0], atom.xyz[1], atom.xyz[2], atom.charge, atom.r_bound))
    solver = ML_Solver()
    solver.load_pqr(pqr_data, type="list")
    solver.epsilon_in = 4.0
    solver.epsilon_out = 80.0
    born_radii = solver.solve()
    for atom, born_radius in zip(background_atoms, born_radii):
        atom.born_radius = born_radius

    # Calculate non-background atoms' Born radii using background atoms as reference
    for res in protein.residue:
        # Include backbone atoms as float
        conf0_atoms = res.conf[0].atom
        for conf in res.conf[1:]: # Start from 1 because we need the float born radii for conformer 1
            print("   Working on conformer %s" % conf.confID)
            # float atoms only
            float_atoms = conf.atom
            pqr_data = [(atom.xyz[0], atom.xyz[1], atom.xyz[2], atom.charge, atom.r_bound) for atom in float_atoms + conf0_atoms]
            solver = ML_Solver()
            solver.load_pqr(pqr_data, type="list")
            solver.epsilon_in = 4.0
            solver.epsilon_out = 80.0
            born_radii = solver.solve()
            for atom, born_radius in zip(float_atoms, born_radii[:len(float_atoms)]):
                atom.float_born_radius = born_radius
            # float atoms in the reference environment
            reference_atoms = [atom for atom in background_atoms if atom.resID != res.resID or atom.confID[3:5] == "BK"]  # Keep background atoms
            pqr_data = [(atom.xyz[0], atom.xyz[1], atom.xyz[2], atom.charge, atom.r_bound) for atom in float_atoms + reference_atoms]
            solver = ML_Solver()
            solver.load_pqr(pqr_data, type="list")
            solver.epsilon_in = 4.0
            solver.epsilon_out = 80.0
            born_radii = solver.solve()
            # Assign born_radius to float_atoms from the first part of born_radii
            for atom, born_radius in zip(float_atoms, born_radii[:len(float_atoms)]):
                atom.born_radius = born_radius

    return


def def_boundary(ir, ic):
    boundary = Exchange(protein)

    boundary.compose_float(protein, ir, ic)
    boundary.compose_single(protein, ir, ic)
    boundary.compose_multi(protein, ir, ic)

    if run_options.debug:
        # Only write the boundary conditions with debug log level
        boundary.write_boundary("float_bnd", "float")
        boundary.write_boundary("single_bnd", "single")
        boundary.write_boundary("multi_bnd", "multi")

    return boundary


def get_conflist(protein):
    conf_list = []
    bkb_list = []
    for res in protein.residue:
        for conf in res.conf:
            if conf.confID[3:5] == "BK":
                bkb_list.append(conf.confID)
            else:
                conf_list.append(conf.confID)
    return conf_list, bkb_list


def safe_pbe(args):
    """
    A wrapper for pbe() to catch exceptions and return error messages.
    """
    try:
        return pbe(args)
    except Exception as e:
        return f"[ERROR] {e}"


def pbe(iric):
    """
    Calculate electrostatic terms: pairwise and reaction filed energy;
    Store results in energies/*.raw file.
    """
    start_time = time.time()

    ir = iric[0]
    ic = iric[1]
    pid = current_process()  # Identify this worker
    confid = protein.residue[ir].conf[ic].confID
    conf_serial = protein.residue[ir].conf[ic].serial
    total_confs = protein.pbe_conformers
    resid = confid[:3] + confid[5:11]
    bound = def_boundary(ir, ic)
    rxn = 0.0

    # skip pbe if atoms in this conformer are all 0 charged
    all_0 = True
    for atom in protein.residue[ir].conf[ic].atom:
        if abs(atom.charge) > 0.001:
            all_0 = False
            break
    if all_0:  # skip
        logger.info("Skipping PBE solver for non-charge conformer (%d/%d) %s..." % (conf_serial, total_confs, confid))
        open(PROGRESS_LOG, "a").write("Skipping PBE solver for non-charge conformer (%d/%d) %s...\n" % (conf_serial, total_confs, confid))
    else:
        # switch to temporary unique directory
        cwd = Path.cwd()
        # use dir + "pbe_data" + cwd (replace / by .) + confid as <tmp_pbe_name>
        tmp_pbe = os.path.join(str(run_options.t), f"pbe_data{cwd.as_posix().replace('/', '.')}_{confid}")
        os.makedirs(tmp_pbe, exist_ok=True)

        try:
            os.chdir(tmp_pbe)

            # decide which pb solver, delphi = delphi legacy
            if run_options.s.upper() == "DELPHI":
                logger.info(
                    "%s: Calling delphi to calculate conformer (%d/%d) %s in %s" % (pid.name, conf_serial, total_confs, confid, tmp_pbe)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling delphi to calculate conformer (%d/%d) %s in %s\n" % (pid.name, conf_serial, total_confs, confid, tmp_pbe))
                pbs_delphi = PBS_DELPHI()
                try:
                    rxn0, rxn = pbs_delphi.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"Delphi run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"

            elif run_options.s.upper() == "ML":
                logger.info(
                    "%s: Calling ML/GB to calculate conformer (%d/%d) %s" % (pid.name, conf_serial, total_confs, confid)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling ML to calculate conformer (%d/%d) %s\n" % (pid.name, conf_serial, total_confs, confid))
                pbs_ml = PBS_ML()
                try:
                    rxn0, rxn = pbs_ml.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"ML run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"

            elif run_options.s.upper() == "NGPB":
                logger.info(
                    "%s: Calling ngpb to calculate conformer (%d/%d) %s in %s" % (pid.name, conf_serial, total_confs, confid, tmp_pbe)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling ngpb to calculate conformer (%d/%d) %s in %s\n" % (pid.name, conf_serial, total_confs, confid, tmp_pbe))
                pbs_ngpb = PBS_NGPB(instance_name)
                try:
                    rxn0, rxn = pbs_ngpb.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"NGPB run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"
                
            elif run_options.s.upper() == "ZAP":
                logger.info(
                    "%s: Calling zap to calculate conformer (%d/%d) %s in %s" % (pid.name, conf_serial, total_confs, confid, tmp_pbe)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling zap to calculate conformer (%d/%d) %s in %s\n" % (pid.name, conf_serial, total_confs, confid, tmp_pbe))
                pbs_zap = PBS_ZAP()
                try:
                    rxn0, rxn = pbs_zap.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"ZAP run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"
                

            elif run_options.s.upper() == "APBS":
                logger.info(
                    "%s: Calling APBS to calculate conformer (%d/%d) %s in %s" % (pid.name, conf_serial, total_confs, confid, tmp_pbe)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling APBS to calculate conformer (%d/%d) %s in %s\n" % (pid.name, conf_serial, total_confs, confid, tmp_pbe))
                pbs_apbs = PBS_APBS()
                try:
                    rxn0, rxn = pbs_apbs.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"APBS run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"

            elif run_options.s.upper() == "TEMPLATE":
                logger.info(
                    "%s: Calling template to calculate conformer (%d/%d) %s in %s" % (pid.name, conf_serial, total_confs, confid, tmp_pbe)
                )
                open(cwd.joinpath(PROGRESS_LOG), "a").write("%s: Calling template to calculate conformer (%d/%d) %s in %s\n" % (pid.name, conf_serial, total_confs, confid, tmp_pbe))
                pbs_template = PBS_TEMPLATE()
                try:
                    rxn0, rxn = pbs_template.run(bound, run_options)
                except Exception as e:
                    logger.critical(f"Template run failed for conformer {confid}: {e}", exc_info=True)
                    return f"[ERROR] {e}"

            else:
                logger.critical("Unknown PBE solver name: %s" % run_options.s)
                sys.exit("Unknown PBE solver name.")

        except Exception as e:
            print("An error occurred:", e)

        finally:
            # Switch back to the original directory
            os.chdir(cwd)
            if not run_options.debug:
                shutil.rmtree(tmp_pbe)

    # write raw opp file
    fname = "%s/%s.raw" % (energy_folder, confid)

    if not Path(energy_folder).exists():
        Path(energy_folder).mkdir()

    if all_0:
        # create an empty file as a marker to indicate this conformer has been calculated
        Path(fname).write_text("")
    else:
        # generate electrostatic interaction raw file
        raw_lines = []

        # Part 1: Method = run_options.s
        line = "[Method] %s\n" % run_options.s
        raw_lines.append(line)

        # Part 2: pw to other conformers, single and multi
        pw_single = defaultdict(float)
        for ia, bound_single in enumerate(bound.single_bnd_xyzrcp):
            # print(confid, bound.single_bnd_atom[ia][0].confID, bound.single_bnd_atom[ia][0].name)
            # single conformer:
            pw_confname = bound.single_bnd_atom[ia][0].confID
            p = bound_single.p * bound.single_bnd_atom[ia][0].charge
            if abs(p) > 0.0001:
                pw_single[pw_confname] += p

        # convert to Kcal and remove interaction within residue including backbone piece
        pw_single.update((key, value / KCAL2KT) for key, value in pw_single.items())
        # print(pw_single)
        # print(len(pw_single))

        pw_multi = defaultdict(float)
        for ia, bound_multi in enumerate(bound.multi_bnd_xyzrcp):
            for atom in bound.multi_bnd_atom[ia]:
                pw_confname = atom.confID
                if pw_confname[3:5] == "BK":
                    continue
                p = bound_multi.p * atom.charge
                if abs(p) > 0.0001:
                    pw_multi[pw_confname] += p


        # convert to Kcal
        pw_multi.update((key, value / KCAL2KT) for key, value in pw_multi.items())
        # for key, value in pw_multi.items():
        #     if abs(value) >= 0.001:
        #         print("%s %8.3f" % (key, value))

        # get conformer list and backbone segment list
        conf_list, bkb_list = get_conflist(protein)

        # write pw section
        line = "\n[PAIRWISE confID single multi flag, kcal/mol]\n"
        raw_lines.append(line)

        for pw_conf in conf_list:
            if resid == pw_conf[:3] + pw_conf[5:11]:
                continue  # skip conformer within residue
            single = 0.0
            multi = 0.0
            non0 = False
            reference = ""
            if pw_conf in pw_single:
                non0 = True
                single = pw_single[pw_conf]
                reference = "*"  # this is marked as a reference conformer for boundary correction
            if pw_conf in pw_multi:
                non0 = True
                multi = pw_multi[pw_conf]
            if non0 and (abs(single) >= PW_CUTOFF or abs(multi) >= PW_CUTOFF):
                line = "%s %8.3f %8.3f %s\n" % (pw_conf, single, multi, reference)
                raw_lines.append(line)

        # Part 3: backbone interaction total
        bkb_total = 0.0
        bkb_breakdown_lines = ["\n[BACKBONE breakdown, kcal/mol]\n"]
        for pw_conf in bkb_list:
            non0 = False
            bkb_pw = 0.0
            if pw_conf in pw_single:
                non0 = True
                bkb_pw = pw_single[pw_conf]
                # exclude the backbone piece when calculating the total
                # if resid != (pw_conf[:3] + pw_conf[5:11]):
                #     bkb_total += bkb_pw
                bkb_total += bkb_pw  # do inclusive calculation for now

            if non0 and abs(bkb_pw) >= PW_CUTOFF:
                line = "%s %8.3f\n" % (pw_conf, bkb_pw)
                bkb_breakdown_lines.append(line)

        line = "\n[BACKBONE total including self, kcal/mol] %8.3f\n" % bkb_total
        raw_lines.append(line)

        # Part 4: backbone interaction breakdown
        line = "\n[BACKBONE breakdown]\n"
        raw_lines += bkb_breakdown_lines

        # Part 5: rxn
        if run_options.fly:
            line = "\n[RXN0, kcal/mol] %8.3f" % rxn0
            raw_lines.append(line)

        line = "\n[RXN, kcal/mol] %8.3f" % rxn
        raw_lines.append(line)

        with open(fname, "w") as ofh:
            ofh.writelines(raw_lines)

    progress_log = "progress.log"

    n_conf = 0
    for res in protein.residue:
        n_conf += len(res.conf) - 1

    end_time = time.time()
    if all_0:
        summary_line = (
            "Skipping conformer %s, no charge, skip ... in %.2f seconds.\n"
            % (confid, end_time - start_time)
        )
    else:
        summary_line = (
            "Computing conformer %s with %s ... finished in %.2f seconds.\n"
            % (confid, run_options.s, end_time - start_time)
        )
    with open(progress_log, "a") as proglog:
        proglog.write(summary_line)

    return (ir, ic)


def is_conf_clash(conf1, conf2, use_r_bound=True):
    """Quick detection of the conformer to conformer clash without considering connectivity.
    If use_r_bound is True, then use the radius of dielectric boundary,
    otherwise use r_vdw, Van der Waals radius
    """
    clash = False
    for atom1 in conf1.atom:
        if clash:
            break
        if use_r_bound:
            r1 = (
                atom1.r_bound * 0.5
            )  # artificially allow less strict clash detection to match old mcce
        else:
            r1 = atom1.r_vdw
        for atom2 in conf2.atom:
            if use_r_bound:
                r2 = (
                    atom2.r_bound * 0.5
                )  # artificially allow less strict clash detection to match old mcce
            else:
                r2 = atom2.r_vdw
            d2 = ddvv(atom1.xyz, atom2.xyz)
            if d2 < (r1 + r2) ** 2:
                clash = True
                # print(atom1.atomID, atom2.atomID, math.sqrt(d2))
                break

    return clash


def postprocess_ele(protein):
    conf_list, _ = get_conflist(protein)
    ele_matrix = {}
    ele_k_s2m_byresid = {}

    # load raw
    for conf in conf_list:
        fname = "%s/%s.raw" % (energy_folder, conf)
        if not Path(fname).is_file():
            continue

        with open(fname) as lines:
            pw_start = False
            resid1 = conf[:3] + conf[5:11]

            for line in lines:
                if line[:9] == "[PAIRWISE":
                    pw_start = True
                    continue
                elif line[:9] == "[BACKBONE":
                    break
                if pw_start:
                    fields = line.split()
                    if len(fields) >= 3:
                        conf2 = fields[0]
                        single = float(fields[1])
                        multi = float(fields[2])
                        if len(fields) > 3:
                            mark = fields[3]
                        else:
                            mark = ""
                        ele_pw = ElePW()
                        ele_pw.multi = multi
                        ele_pw.single = single
                        ele_pw.mark = mark
                        ele_matrix[(conf, conf2)] = ele_pw

                        if "*" in mark:
                            resid2 = conf2[:3] + conf2[5:11]
                            if abs(multi) > 0.1:
                                k_single_multi = single / multi
                            else:
                                k_single_multi = 1.0
                            if k_single_multi > 1.0:
                                k_single_multi = 1.0
                            ele_k_s2m_byresid[(resid1, resid2)] = k_single_multi

    # scale multi by the k factor, or use single for reference point
    for conf_pair, ele_pw in ele_matrix.items():
        conf1, conf2 = conf_pair
        resid1 = conf1[:3] + conf1[5:11]
        resid2 = conf2[:3] + conf2[5:11]
        key = (resid1, resid2)
        if key in ele_k_s2m_byresid:
            k = ele_k_s2m_byresid[(resid1, resid2)]
        else:
            k = 1.0
        if "*" in ele_pw.mark:
            ele_pw.scaled = ele_pw.single  # use the reference directly
        else:
            ele_pw.scaled = k * ele_pw.multi

    # for conf_pair, ele_pw in ele_matrix.items():
    #     print("%s: %8.3f %8.3f %8.3f %s" % (conf_pair, ele_pw.scaled, ele_pw.single, ele_pw.multi, ele_pw.mark))

    # Find clashing dielectric boundary. This is because the single conformation boundary is composed by the selected
    # conformer + the native conformer of other residues. This selected conformer may have geometry conflict with the
    # native conformers. If a clash is identified, a question mark will be added to the ele_pw mark.
    logger.debug("Detecting conformer to conformer clashes ...")
    for res1 in protein.residue:
        if len(res1.conf) > 1:  # skip dummy or backbone only residue
            for conf1 in res1.conf[1:]:
                conf1_id = conf1.confID
                for res2 in protein.residue:
                    if res2 == res1:
                        continue
                    # find the reference conformer to res2
                    conf2_ref = None
                    if len(res2.conf) > 1:  # skip dummy or backbone only residue
                        for conf2 in res2.conf[1:]:
                            conf2_id = conf2.confID
                            key = (conf1_id, conf2_id)
                            if key in ele_matrix:
                                ele_pw = ele_matrix[key]
                                if "*" in ele_pw.mark:
                                    conf2_ref = conf2
                                    break

                    # detect the clash between conf1 and conf2_ref
                    clash = False
                    if conf2_ref:
                        clash = is_conf_clash(conf1, conf2_ref)

                    # mark res2 conformers with "?"
                    if clash:
                        # print(conf1.confID, conf2_ref.confID)
                        for conf2 in res2.conf[1:]:
                            conf2_id = conf2.confID
                            key = (conf1_id, conf2_id)
                            if key in ele_matrix:
                                ele_pw = ele_matrix[key]
                                ele_pw.mark = "?" + ele_pw.mark

    # ele correction and average
    # The correction starts with ele_pw.scaled, and follow the following rules to make correction
    # C-C: Charged to charged abnormal case: reduce by a factor of 1.5 of the smaller pw at multi condition
    # C-C: Charged to charged normal case: average scaled pws
    # D-D: dipole to dipole, reduce by a factor of 2.0
    # C-D: charged to dipole,
    for conf_pair, ele_pw in ele_matrix.items():
        if conf_pair[0][3] == "+" or conf_pair[0][3] == "-":
            conf1_type = "C"
        else:
            conf1_type = "D"
        if conf_pair[1][3] == "+" or conf_pair[1][3] == "-":
            conf2_type = "C"
        else:
            conf2_type = "D"
        reversed_key = (conf_pair[1], conf_pair[0])
        if reversed_key in ele_matrix:
            reversed_ele_pw = ele_matrix[reversed_key]
        else:
            reversed_ele_pw = ElePW()
        if conf1_type == "C" and conf2_type == "C":
            # abnormal case, opposite sign when scaled, or both directions have ? mark,
            # scaled down by factor 1.5 from the smaller of the two multi
            if (
                ele_pw.scaled * ele_pw.multi < 0
                or reversed_ele_pw.scaled * reversed_ele_pw.multi < 0
                or ("?" in ele_pw.mark and "?" in reversed_ele_pw.mark)
            ):
                if abs(ele_pw.multi) < abs(reversed_ele_pw.multi):
                    ele_pw.averaged = ele_pw.multi / 1.5
                    reversed_ele_pw.averaged = ele_pw.multi / 1.5
                else:
                    ele_pw.averaged = reversed_ele_pw.multi / 1.5
                    reversed_ele_pw.averaged = reversed_ele_pw.multi / 1.5

                # print("%s -> %s    |    %s -> %s" % (conf_pair[0], conf_pair[1],
                #                                      reversed_key[0], reversed_key[1]))
                # print("Multi:        %8.3f |    %8.3f" % (ele_matrix[conf_pair].multi,
                #                                           ele_matrix[reversed_key].multi))
                # print("Single:       %8.3f |    %8.3f" % (ele_matrix[conf_pair].single,
                #                                           ele_matrix[reversed_key].single))
                # print("Scaled:       %8.3f |     %8.3f" % (ele_matrix[conf_pair].scaled,
                #                                            ele_matrix[reversed_key].scaled))
                # print("Averaged:     %8.3f |     %8.3f" % (ele_matrix[conf_pair].averaged,
                #                                            ele_matrix[reversed_key].averaged))
                # print(vars(ele_pw))
                # print(vars(reversed_ele_pw))
            elif "?" in ele_pw.mark and "?" not in reversed_ele_pw.mark:
                ele_pw.averaged = reversed_ele_pw.scaled
                reversed_ele_pw.averaged = reversed_ele_pw.scaled
            elif "?" not in ele_pw.mark and "?" in reversed_ele_pw.mark:
                ele_pw.averaged = ele_pw.scaled
                reversed_ele_pw.averaged = ele_pw.scaled
            else:
                if abs(ele_pw.scaled) < abs(reversed_ele_pw.scaled):
                    ele_pw.averaged = reversed_ele_pw.averaged = ele_pw.scaled
                else:
                    ele_pw.averaged = reversed_ele_pw.averaged = reversed_ele_pw.scaled
        elif conf1_type == "D" and conf2_type == "D":
            ele_pw.averaged = reversed_ele_pw.averaged = (
                ele_pw.scaled + reversed_ele_pw.scaled
            ) / 2.0
            #  Originally averaging multi  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        else:  # must be "D" to "C"
            if (
                ele_pw.scaled * ele_pw.multi < 0
                or reversed_ele_pw.scaled * reversed_ele_pw.multi < 0
                or ("?" in ele_pw.mark and "?" in reversed_ele_pw.mark)
            ):
                ele_pw.averaged = reversed_ele_pw.averaged = (
                    (ele_pw.multi + reversed_ele_pw.multi) / 2.0 / 1.5
                )
            elif "?" in ele_pw.mark and "?" not in reversed_ele_pw.mark:
                ele_pw.averaged = reversed_ele_pw.averaged = reversed_ele_pw.scaled
            elif "?" not in ele_pw.mark and "?" in reversed_ele_pw.mark:
                ele_pw.averaged = reversed_ele_pw.averaged = ele_pw.scaled
            else:
                ele_pw.averaged = reversed_ele_pw.averaged = (
                    ele_pw.scaled + reversed_ele_pw.scaled
                ) / 2.0

    return ele_matrix


def compose_opp(protein, ele_matrix):

    epath = "energies"
    for res1 in protein.residue:
        for conf1 in res1.conf[1:]:
            master1_confID = conf1.confID[:10]+"_"+conf1.confID[11:]            
            master_raw_file = "%s/%s.raw" % (epath, master1_confID)
            if not Path(master_raw_file).is_file():
                continue
            # only create opp files when a raw file exists
            fname = "%s/%s.opp" % (epath, conf1.confID)
            lines = []
            for res2 in protein.residue:
                if res2 != res1:
                    for conf2 in res2.conf[1:]:
                        master2_confID = conf2.confID[:10]+"_"+conf2.confID[11:]
                        conf_pair = (master1_confID, master2_confID)
                        if conf_pair in ele_matrix:
                            average = ele_matrix[conf_pair].averaged
                            scaled = ele_matrix[conf_pair].scaled
                            multi = ele_matrix[conf_pair].multi
                            mark = ele_matrix[conf_pair].mark
                        else:
                            average = scaled = multi = 0.0
                            mark = ""

                        if (conf1.confID, conf2.confID) in protein.vdw_pw and \
                            abs(protein.vdw_pw[(conf1.confID, conf2.confID)]) > PW_CUTOFF:
                            vdw = protein.vdw_pw[(conf1.confID, conf2.confID)]
                        else:
                            vdw = 0.0
                        if vdw > PW_CUTOFF or abs(average) > PW_CUTOFF:
                            lines.append(
                                "%05d %s %8.3f %7.3f %7.3f %7.3f %s\n"
                                % (
                                    conf2.serial,
                                    conf2.confID,
                                    average,
                                    vdw,
                                    scaled,
                                    multi,
                                    mark,
                                )
                            )
            with open(fname, "w") as opp:
                opp.writelines(lines)

    return


def compose_head3(protein):
    epath = "energies"
    h3_hdr = (
        "iConf CONFORMER     FL  occ    crg   Em0  pKa0 ne nH    "
        "vdw0    vdw1    tors    epol   dsolv   extra    history\n"
    )
    head3lines = [h3_hdr]
    # read backbone ele interaction epol
    epol_all = {}
    rxn_all = {}
    rxn0_all = {}
    for res in protein.residue:
        for conf in res.conf[1:]:
            master_confID = conf.confID[:10] + "_" + conf.confID[11:]
            fname = "%s/%s.raw" % (epath, master_confID)
            if os.path.isfile(fname):
                lines = open(fname).readlines()
                for line in lines:
                    if line.startswith("[BACKBONE total"):
                        fields = line.split("]")
                        epol_all[conf.confID] = float(fields[-1])
                    elif line.startswith("[RXN0,"):
                        fields = line.split("]")
                        rxn0_all[conf.confID] = float(fields[-1])
                    elif line.startswith("[RXN,"):
                        fields = line.split("]")
                        rxn_all[conf.confID] = float(fields[-1])

    # natom dictonary to determine dummy conformers
    natom_byconftype = {}
    for key, value in env.param.items():
        if key[0] == "CONFLIST":
            for conftype in env.param[("CONFLIST", key[1])]:
                natom_byconftype[conftype] = 0

    for key, value in env.param.items():
        if key[0] == "CONNECT":
            conftype = key[2]
            if conftype in natom_byconftype:
                # there are cases CONNECT exists but conftype was commented out
                natom_byconftype[conftype] += 1

    serial = 1
    for res in protein.residue:
        # add dummy conformers
        for conftype in env.param[("CONFLIST", res.resID[:3])]:
            natom = natom_byconftype[conftype]
            if natom == 0 and conftype[-2:] != "BK":
                newconf = Conformer()
                n = len(res.conf)
                newconf.confID = "%s%s%s%s_%03d" % (
                    res.resID[:3],
                    conftype[-2:],
                    res.resID[7],
                    res.resID[3:7],
                    n,
                )
                newconf.history = "DM"
                newconf.mark = "d"
                res.conf.append(newconf)

        # tors_confs = []
        # for conf in res.conf[1:]:
        #     tors_confs.append(torsion(conf))
        # if tors_confs:
        #     min_tors = min(tors_confs)
        #     tors_confs = [x - min_tors for x in tors_confs]
        count = 0
        for conf in res.conf[1:]:
            # FIX: local variable 'iconf' is assigned to but never used
            # if conf.mark == "d":
            #     iconf = 0
            # else:
            #     iconf = conf.i + 1
            confID = conf.confID            
            flag = "f"
            occ = 0.0
            crg = conf.crg
            conftype = conf.confID[:5]
            query_key = ("CONFORMER", conftype)
            if query_key in env.param:
                value = env.param[query_key]
                em0 = value.param["em0"]
                pka0 = value.param["pka0"]
                ne = value.param["ne"]
                nh = value.param["nh"]
            else:
                logger.warning(
                    (
                        f"CONFORMER record of {conftype} not defined in ftpl file; "
                        "em0, pka0, ne and nh are assumed to be 0"
                    )
                )
                em0 = 0.0
                pka0 = 0.0
                ne = 0
                nh = 0
            vdw0 = conf.vdw0
            vdw1 = conf.vdw1
            # tors = tors_confs[count]
            tors = conf.tors
            count += 1

            if confID in epol_all:
                epol = epol_all[conf.confID]
            else:
                epol = 0.0

            if confID in rxn_all:
                rxn = rxn_all[conf.confID]
            else:
                rxn = 0.0

            if run_options.fly:
                if conf.confID in rxn0_all:
                    rxn0 = rxn0_all[conf.confID]
                else:
                    rxn0 = 0.0
            else:
                epsilon = run_options.d
                key3 = "rxn%02d" % int(epsilon)
                if ("CONFORMER", conftype) in env.param:
                    rxn0 = env.param["CONFORMER", conftype].param[key3]
                else:
                    rxn0 = 0.0
                    logger.warning(
                        (
                            f"CONFORMER record of {conftype} not defined "
                            "in ftpl file. rxn0 is assumed to be 0"
                        )
                    )
            dsolv = rxn - rxn0
            history = conf.history

            key = ("EXTRA", conftype)
            if key in env.param:
                extra = env.param["EXTRA", conftype]
            else:
                extra = 0.0

            if Path("%s/%s.raw" % (epath, conf.confID)).is_file():
                # only create opp files when a raw file exists
                mark = "t"
            elif confID[10] in {"+", "-"}:
                mark = "v"  # vdw virtual conformer
            else:
                mark = "f"
            if conf.mark == "d":
                mark = conf.mark  # inherit dummy mark from conformer

            head3lines.append(
                (
                    "%05d %s %s %4.2f %6.3f %5d %5.2f %2d %2d "
                    "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %10s %s\n"
                )
                % (
                    serial,
                    confID,
                    flag,
                    occ,
                    crg,
                    em0,
                    pka0,
                    ne,
                    nh,
                    vdw0,
                    vdw1,
                    tors,
                    epol,
                    dsolv,
                    extra,
                    history,
                    mark,
                )
            )

            conf.serial = serial
            serial += 1

    with open("head3.lst", "w") as h3:
        h3.writelines(head3lines)

    return


def cli_parser():
    helpmsg = "Run mcce step 3, energy lookup table calculations."
    parser = argparse.ArgumentParser(description=helpmsg)
    parser.add_argument(
        "-c",
        metavar=("start", "end"),
        default=[1, 99999],
        nargs=2,
        help="starting and ending conformer; default: 1 99999",
        type=int,
    )
    parser.add_argument(
        "-d",
        metavar="epsilon",
        default="4.0",
        help="protein dielectric constant; default: %(default)s.",
    )
    parser.add_argument(
        "-do",
        metavar="dielectric_outside",
        default="80.0",
        help="outside material dielectric constant; default: %(default)s.",
    )
    parser.add_argument(
        "-s",
        metavar="pbs_name",
        default="ngpb",
        choices=["delphi", "ngpb", "zap", "apbs", "ml", "template"],
        help="PBE solver; choices: ngpb, delphi, zap, ml; default: %(default)s.",
    )
    parser.add_argument(
        "-t",
        metavar="tmp_folder",
        default="/tmp",
        help="PBE solver temporary folder; You need to make this directory writable for all (chmod a+w) when using ngbp; default: %(default)s.",
    )
    parser.add_argument(
        "-p",
        metavar="processes",
        type=int,
        default=1,
        help="run step 3 using p processes; default: %(default)s.",
    )
    parser.add_argument(
        "-ftpl",
        metavar="ftpl_folder",
        default="",
        help="ftpl folder; default: 'param/' of mcce executable location."
    )
    parser.add_argument(
        "-salt",
        metavar="salt_concentration",
        default=0.15,
        type=float,
        help="Salt concentration in moles/L; default: %(default)s.",
    )
    parser.add_argument(
        "--skip_pb", default=False, action="store_true",
        help="run vdw and torsion calculation only; default: %(default)s.",
    )
    parser.add_argument(
        "--old_vdw", default=False, action="store_true",
        help="Run old vdw function calculations; default: %(default)s.",
    )    
    parser.add_argument(
        "-vdw_relax",
        metavar="vdw_R_relaxation",
        default=0,
        type=float,
        help="relax vdw R parameter by +- specified value; default: %(default)s.",
    )
    parser.add_argument(
        "--fly", default=False, action="store_true",
        help="don-the-fly rxn0 calculation; default: %(default)s.",
    )
    parser.add_argument(
        "--refresh",
        default=False,
        action="store_true",
        help="recreate *.opp and head3.lst from step2_out.pdb and *.raw files; default: %(default)s.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="print debug information and keep pbe solver tmp; default: %(default)s.",
    )
    parser.add_argument(
        "-l", metavar="file", default="",
        help="load above options from a file; default: %(default)s.",
    )
    parser.add_argument(
        "-u",
        metavar="Key=Value",
        default="",
        help="User customized variables, overwrites run.prm; default: %(default)s.",
    )
    parser.add_argument(
        "-load_runprm",
        metavar="prm_file",
        default="",
        help="Load additional run.prm file, overwrite default values; default: %(default)s.",
    )

    return parser


def sanitycheck_step3(t0):
    """
    Sanity check for step 3. 
    
    Input t0: start time of step 3 run
    Returns:
        STATUS: messages
        STATUS = SUCCESS, WARNING, ERROR
    """
    status = "SUCCESS"
    messages = []

    # Check head3.lst
    head3_file = "head3.lst"
    confids = []
    if not Path(head3_file).is_file():
        status = "ERROR"
        messages.append(f" No {head3_file} detected, step 3 failed.")
    elif os.path.getmtime(head3_file) < t0:
        status = "ERROR"
        messages.append(f" {head3_file} was not updated during this run, step 3 failed")
    else:
        # Check the flag column c[16] of head3.lst. It should be 't' for all non-dummy conformers.
        # Dummy conformers have mark 'd' in column c[80]
        partial_run = False
        with open(head3_file) as h3:
            lines = h3.readlines()[1:]  # skip header line
            for line in lines:
                parts = line.split()
                confID = parts[1].strip()
                flag = parts[16].strip()
                if flag != "d" and flag != "t":
                    partial_run = True
                    break
                else:
                    if flag == "t":
                        confids.append(confID)
        if partial_run:    
            status = "WARNING"
            messages.append(" head.lst suggests a partial step 3 run")   # rewrite the last message

    # Check .opp files, existence and time stamp                
    if confids:
        opp_uptodate = True
        opp_present = True
        for confID in confids:
            opp_file = f"energies/{confID}.opp"
            if not os.path.isfile(opp_file):
                opp_present = False
                print(f"Missing {opp_file}")
                status = "ERROR"
                break
            elif os.path.getmtime(opp_file) < t0:
                opp_uptodate = False
                print(t0, os.path.getmtime(opp_file))
                print(f"Outdated {opp_file}")
                if status != "ERROR":  # ERROR state has higher priority
                    status = "WARNING"
                break
        if not opp_present:
            messages.append(" Some .opp files are missing.")
        if not opp_uptodate:
            messages.append(" Some .opp files are outdated, it is normal if this is a partial run.")

    # Final report
    if status == "SUCCESS":
        messages.append("Step 3 completed successfully.")
    else:
        messages.append(" Check the log to see what went wrong.")
    msg_str = f"[{status}]: " + ";".join(messages)

    return msg_str


if __name__ == "__main__":
    t0 = time.time()

    parser = cli_parser()
    args = parser.parse_args()

    log_level = "INFO"
    if args.debug:
        log_level = "DEBUG"
    # config logging to screen, 'run.log' and 'err.log', and maybe 'step3.debug';
    logger = config_logger(step_num=3, log_level=log_level)

    logger.info("Step 3. Energies calculation with PBE solver.")
    if "--old_vdw" in sys.argv:
         logger.info("Calling old VDW....Scaling Applied")
    else:
         logger.info("Calling new VDW....No Scaling Applied")

    start_t0 = start_t = time.time()

    logger.info("   Process run time options & convert step2_out.pdb.")
    run_options = RunOptions(args)
    if run_options.s.upper() == "ML":
        run_options.fly = True  # force fly to be true for ML

    # print(vars(run_options))

    # environment and ftpl
    detected = detect_runprm()
    env.load_runprm()
    if run_options.ftpl:
        env.runprm["FTPLDIR"] = run_options.ftpl
    else:
        env.runprm["FTPLDIR"] = str(Path(__file__).parent.parent/"param")
    env.load_ftpl()
    # env.print_param()

    logger.info("   Prepare input for PB solver.")
    protein = Protein()
    protein.loadpdb(run_options.inputpdb)
    protein.update_confcrg()
    if args.s.upper() == "ML":
        logger.info("   Using machine learning model as PBE solver, calculating born radius.")
        calculate_born_radius(protein)
        # Save results to a file
        with open("born_radii.csv", "w") as f:
            header = [f"CONFID,NAME,X,Y,Z,CHARGE,BOUND_RADIUS,FLOAT_BORN_RADIUS,BORN_RADIUS\n"]
            f.writelines(header)
            for res in protein.residue:
                for conf in res.conf:
                    for atom in conf.atom:
                        f.write(f"{conf.confID},\"{atom.name}\",{atom.xyz[0]:.3f},{atom.xyz[1]:.3f},{atom.xyz[2]:.3f},{atom.charge:.3f},{atom.r_bound:.3f},{atom.float_born_radius:.3f},{atom.born_radius:.3f}\n")
    start_t = time.time()

    if run_options.s.upper() == "NGPB":
        bind_path = run_options.t 
        if os.path.commonpath([bind_path, "/tmp"]) == "/tmp":
            parent_dir = bind_path
        else:
            parent_dir = os.path.dirname(bind_path)

        instance_name = f"ngpb_{os.getpid()}_{int(time.time())}"
        # Find the container path (relies on the .sif file being executable)
        container_path = shutil.which("NextGenPB_MCCE4.sif")
        try:
            # Create an instance for this submission:
            result = subprocess.run(
                    f"apptainer instance start --bind {parent_dir}:{parent_dir} {container_path} {instance_name}",
                    shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start apptainer instance: {e.stderr.decode().strip()}")
            # Does not matter, processing goes on with, possibly an instance "bound" to a
            # different parent_dir??

    if not args.skip_pb:
        # Prepare input for PB solver:
        #   common_boundary, sites to receive potential, and PB conditions;
        #   make conformer list with their corresponding ir and ic.
        #   This list of (ir, ic) will be passed as an array to the multiprocess module as work load.

        # Serialize conformer index
        serial_counter = 1
        for ir in range(len(protein.residue)):
            if len(protein.residue[ir].conf) > 1:  # skip dummy or backbone only residue
                for ic in range(1, len(protein.residue[ir].conf)):
                    protein.residue[ir].conf[ic].serial = serial_counter
                    serial_counter += 1

        work_load = []
        counter = 1
        for ir in range(len(protein.residue)):
            if len(protein.residue[ir].conf) > 1:  # skip dummy or backbone only residue
                for ic in range(1, len(protein.residue[ir].conf)):
                    if run_options.end >= counter >= run_options.start:
                        work_load.append((ir, ic))
                    counter += 1
        protein.pbe_conformers = counter - 1

        logger.debug("  work_load as (ir ic) list [%s]" % ",".join(map(str, work_load)))
        logger.info("   Time needed: %d seconds.", time.time() - start_t)
        start_t = time.time()

        logger.info("   Set up parallel envrionment and run PB solver.")
        max_pool = run_options.p
        logger.info("   Running PBE solver in %d threads" % max_pool)

#        with Pool(max_pool) as pool:
#            results = []
#            async_results = [pool.apply_async(safe_pbe, (arg,)) for arg in work_load]
#
#            for i, r in enumerate(async_results):
#                try:
#                    res = r.get(timeout=300)  # 300 seconds timeout per task, test if this solves premature termination
#                except TimeoutError:
#                    logger.error(f"Task {i} timed out")
#                    results.append("[ERROR] Task timed out")
#                    pool.terminate()   # stop all workers immediately
#                    exit()              # exit to stop program
#                except Exception as e:
#                    logger.error(f"Task {i} failed with error: {e}")
#                    results.append("[ERROR] Task failed")

        failed_tasks = []
        with Pool(max_pool) as pool:
            async_results = [pool.apply_async(safe_pbe, (arg,)) for arg in work_load]
            for i, r in enumerate(async_results):
                ir, ic = work_load[i]
                confid = protein.residue[ir].conf[ic].confID
                try:
                    res = r.get(timeout=PBE_TASK_TIMEOUT)
                except TimeoutError:
                    logger.error(f"Task {i} ({confid}) timed out after {PBE_TASK_TIMEOUT}s")
                    failed_tasks.append((i, confid, "timeout"))
                    continue
                except Exception as e:
                    logger.error(f"Task {i} ({confid}) failed: {e}")
                    failed_tasks.append((i, confid, str(e)))
                    continue

        if failed_tasks:
            with open("failed_conformers.txt", "w") as f:
                for i, cid, reason in failed_tasks:
                    f.write(f"{i}\t{cid}\t{reason}\n")
            logger.warning(f"{len(failed_tasks)} conformers failed; see failed_conformers.txt")

        cwd = os.getcwd()
        logger.info("   Time needed: %d seconds.", time.time() - start_t)

        logger.info("   Post-processing of electrostatic potential.")
        start_t = time.time()

    if run_options.s.upper() == "NGPB":
        subprocess.run(["apptainer", "instance", "stop", instance_name], check=False)

        
    logger.info("   Processing ele pairwise interaction...")
    ele_matrix = postprocess_ele(protein)
    # Debug
    # for conf_pair, ele_pw in ele_matrix.items():
    #     reversed_key = (conf_pair[1], conf_pair[0])
    #     ele_pw = ele_matrix[conf_pair]
    #     if reversed_key in ele_matrix:
    #         reversed_ele_pw = ele_matrix[reversed_key]
    #     else:
    #         reversed_ele_pw = ElePW()
    #
    #     if abs(ele_pw.multi) < 0.1:
    #         continue
    #     print("%s -> %s    |    %s -> %s" % (conf_pair[0], conf_pair[1], reversed_key[0], reversed_key[1]))
    #     print("Multi:        %8.3f              |          %8.3f" % (ele_pw.multi, reversed_ele_pw.multi))
    #     print("Single:       %8.3f              |          %8.3f" % (ele_pw.single, reversed_ele_pw.single))
    #     print("Scaled:       %8.3f              |          %8.3f" % (ele_pw.scaled, reversed_ele_pw.scaled))
    #     print("Averaged:     %8.3f              |          %8.3f" % (ele_pw.averaged, reversed_ele_pw.averaged))
    #     print("Mark:         %8s              |          %8s" % (ele_pw.mark, reversed_ele_pw.mark))

    logger.info("   Time needed: %d seconds.", time.time() - start_t)
    start_t = time.time()

    # Compute vdw, not doing parallelization at this moment
    logger.info("   Making atom connectivity ...")
    protein.make_connect12()
    protein.make_connect13()
    protein.make_connect14()
    logger.info("   Time needed: %d seconds.", time.time() - start_t)
    start_t = time.time()

    logger.info("   Calculating vdw ...")
    # Call subroutine that creates vdw virtual conformers:
    protein.calc_vdw_virtual(delta=args.vdw_relax)
    # For efficiency reason, the vdw pairwise table is a matrix: protein.vdw_pw[conf1.i,conf2.i]
    logger.info("   Time needed: %d seconds.", time.time() - start_t)
    start_t = time.time()

    logger.info("   Calculating torsion energy ...")
    protein.calc_tors()
    logger.info("   Time needed: %d seconds.", time.time() - start_t)
    start_t = time.time()

    if args.vdw_relax:
        # One assumption: these virtual conformers were already made by cal_vdw_virtual
        logger.info("   Calculating torsion energy for VDW virtual conformers ...")
        protein.calc_tors_virtual()
        logger.info("   Time needed: %d seconds.", time.time() - start_t)
        start_t = time.time()

    # Assemble output files, order sensitive as head3.lst subroutine will make serial for
    # conformers later used by opp files
    logger.info("   Composing head3.lst ...")
    compose_head3(protein)
    logger.info("   Time needed: %d seconds.", time.time() - start_t)
    start_t = time.time()

    logger.info("   Composing opp files ...")
    compose_opp(protein, ele_matrix)
    logger.info("   Time needed: %d seconds.", time.time() - start_t)

    # Clean up ligated pw if any
    reset_ligated_pw()

    logger.info("   Total time for step3 is %d seconds.", time.time() - start_t0)

    if detected:
        restore_runprm()

    # Sanity check
    sanity = sanitycheck_step3(t0)
    print(sanity)    
