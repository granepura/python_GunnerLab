#!/usr/bin/env python

"""
Module: step4.py

This program set up and run MCCE step 4.

Input:
 * energies/
 * head3.lst

Output:
 * fort.38
 * pK.out

Usage examples:

1. Run step 4 with default (pH titration from 0.0 to 14.0, without entropy correction)
    step4.py

2. Write run.prm for step 4, do not actually run step 4. (Dry run)
    step4.py --norun

3. Run step 4 at defined points
    step4.py -i 4.0 -d 1 -n 5

4. Run step 4 with entropy correction and 'ms_out' directory retention
    step4.py --xts --ms

5. Run step 4 using specific mcce executable
    step4.py -e /path/to/mcce

6. Run step 4 using Eh titration
    step4.py -t eh

7. Run step 4 with other customized parameters
    step4.py -u EXTRA=./extra.tpl

8. Run step 4 in parallel across pH points using 15 processors
    step4.py -p 15
"""

import argparse
import os
import subprocess
import shutil
import time
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from mccesteps import export_runprm
from mccesteps import record_runprm
from mccesteps import detect_runprm
from mccesteps import restore_runprm
from amend_sumcrg import amend_sum_crg

def build_runprm_dict(args):
    """Build the run.prm parameter dictionary from command-line args."""
    runprm = {}
    path = str(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.dirname(path)
    runprm["MCCE_HOME"] = base_path
    runprm["DO_MONTE"] = "t"
    runprm["EXTRA"] = "%s/extra.tpl" % base_path
    runprm["TITR_TYPE"] = args.t.lower()
    if runprm["TITR_TYPE"] == "ph":
        runprm["TITR_PH0"] = args.i
        runprm["TITR_EH0"] = "0.0"
    elif runprm["TITR_TYPE"] == "eh":
        runprm["TITR_PH0"] = "7.0"
        runprm["TITR_EH0"] = args.i
    runprm["TITR_PHD"] = args.d
    runprm["TITR_EHD"] = args.d
    runprm["TITR_STEPS"] = args.n
    runprm["BIG_PAIRWISE"] = "5.0"
    runprm["MONTE_SEED"] = "-1"
    runprm["MONTE_T"] = "298.15"
    runprm["MONTE_FLIPS"] = "3"
    runprm["MONTE_NSTART"] = "100"
    runprm["MONTE_NEQ"] = "300"
    runprm["MONTE_REDUCE"] = "0.001"
    runprm["MONTE_RUNS"] = "6"
    runprm["MONTE_NITER"] = "2000"
    runprm["MONTE_TRACE"] = "50000"
    runprm["NSTATE_MAX"] = "1000000"
    runprm["MONTE_TSX"] = "t" if args.xts else "f"
    runprm["MFE_POINT"] = "f"
    runprm["MFE_CUTOFF"] = "-1.0"
    runprm["MS_OUT"] = "t" if args.ms else "f"
    if args.u:
        for field in args.u.split(","):
            try:
                key, value = field.split("=")
                runprm[key] = value
            except ValueError:
                print(f"Each component format is 'KEY=VALUE'. Unrecognized: {field}.")
    if args.load_runprm:
        lines = open(args.load_runprm)
        for line in lines:
            entry_str = line.strip().split("#")[0]
            fields = entry_str.split()
            if len(fields) > 1:
                key_str = fields[-1]
                if key_str[0] == "(" and key_str[-1] == ")":
                    key = key_str.strip("()").strip()
                    value = fields[0]
                    runprm[key] = value
    runprm["DO_PREMCCE"] = "f"
    runprm["DO_ROTAMERS"] = "f"
    runprm["DO_ENERGY"] = "f"
    runprm["DO_MONTE"] = "t"
    return runprm


def write_runprm(args):
    runprm = build_runprm_dict(args)
    export_runprm(runprm)
    record_runprm(runprm, "#STEP4")
    return runprm


def run_single_point(mcce_exe, base_dir, temp_dir, runprm):
    """Run mcce for a single titration point in a temp directory."""
    os.makedirs(temp_dir, exist_ok=True)

    head3_src = os.path.join(base_dir, "head3.lst")
    energies_src = os.path.join(base_dir, "energies")
    head3_dst = os.path.join(temp_dir, "head3.lst")
    energies_dst = os.path.join(temp_dir, "energies")

    if not os.path.exists(head3_dst):
        os.symlink(os.path.abspath(head3_src), head3_dst)
    if not os.path.exists(energies_dst):
        os.symlink(os.path.abspath(energies_src), energies_dst)

    runprm_lines = [
        "# WARRPER GENERATED run.prm at %s\n" % datetime.now().strftime("%Y/%m%d, %H:%M:%S")
    ]
    for key in runprm:
        runprm_lines.append("%-20s    (%s)\n" % (runprm[key], key))
    with open(os.path.join(temp_dir, "run.prm"), "w") as f:
        f.writelines(runprm_lines)

    titr_type = runprm["TITR_TYPE"]
    if titr_type == "ph":
        point_val = float(runprm["TITR_PH0"])
    else:
        point_val = float(runprm["TITR_EH0"])

    result = subprocess.run(
        [mcce_exe],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    fort38_path = os.path.join(temp_dir, "fort.38")
    if not os.path.isfile(fort38_path):
        print(f"   WARNING: fort.38 not generated for {titr_type}={point_val}")
        return None

    return temp_dir


def merge_fort38(temp_dirs, titr_type, titr_values):
    """Merge single-column fort.38 files into a combined fort.38 in the current directory."""
    conformer_occ = {}
    conformer_order = []

    for idx, tdir in enumerate(temp_dirs):
        fort38_path = os.path.join(tdir, "fort.38")
        if not os.path.isfile(fort38_path):
            continue
        with open(fort38_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 2:
                continue
            conf_id = fields[0]
            occ_val = float(fields[1])
            if conf_id not in conformer_occ:
                conformer_occ[conf_id] = [0.0] * len(temp_dirs)
                conformer_order.append(conf_id)
            conformer_occ[conf_id][idx] = occ_val

    with open("fort.38", "w") as f:
        if titr_type == "ph":
            f.write(" ph           ")
            for val in titr_values:
                f.write(" %5.1f" % val)
        else:
            f.write(" eh           ")
            for val in titr_values:
                f.write(" %5.0f" % val)
        f.write("\n")
        for conf_id in conformer_order:
            f.write("%s" % conf_id)
            for val in conformer_occ[conf_id]:
                f.write(" %5.3f" % val)
            f.write("\n")


def merge_mc_out(temp_dirs, titr_type, titr_values):
    """Concatenate mc_out files from temp directories."""
    with open("mc_out", "w") as fout:
        for idx, tdir in enumerate(temp_dirs):
            mc_path = os.path.join(tdir, "mc_out")
            if os.path.isfile(mc_path):
                fout.write("=" * 60 + "\n")
                fout.write(f"  {titr_type} = {titr_values[idx]:.1f}\n")
                fout.write("=" * 60 + "\n")
                with open(mc_path) as fin:
                    fout.write(fin.read())


def generate_pK_out(titr_type="ph"):
    """Generate pK.out from fort.38 using sigmoid curve fitting."""
    from scipy.optimize import minimize

    if not os.path.isfile("fort.38"):
        print("WARNING: fort.38 not found; cannot generate pK.out")
        return

    with open("fort.38") as f:
        lines = f.readlines()

    fields = lines[0].split()
    titr_values = [float(x) for x in fields[1:]]
    n_points = len(titr_values)

    conf_occ = {}
    conf_order = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != n_points + 1:
            continue
        conf_id = fields[0]
        occ = [float(v) for v in fields[1:]]
        conf_occ[conf_id] = occ
        conf_order.append(conf_id)

    res_occ = {}
    res_order = []
    for conf_id in conf_order:
        if conf_id[3] not in ('+', '-'):
            continue
        res_key = conf_id[:4] + conf_id[5:11]
        if res_key not in res_occ:
            res_occ[res_key] = [0.0] * n_points
            res_order.append(res_key)
        for j in range(n_points):
            res_occ[res_key][j] += conf_occ[conf_id][j]

    def sigmoid(x, a, b):
        t = a * (x - b)
        t = max(min(t, 500), -500)
        e = math.exp(t)
        return e / (1.0 + e)

    def score(params, xp, yp):
        a, b = params
        s = 0.0
        for i in range(len(xp)):
            yt = sigmoid(xp[i], a, b)
            s += (yt - yp[i]) ** 2
        return s

    with open("pK.out", "w") as fp:
        if titr_type == "ph":
            fp.write("  pH      ")
        else:
            fp.write("  Eh      ")
        fp.write("       pKa/Em  n(slope) 1000*chi2\n")

        for res_key in res_order:
            yp = res_occ[res_key]

            mid_guess = titr_values[n_points // 2]
            best_mid = 0.5
            for j in range(n_points):
                if abs(yp[j] - 0.5) < best_mid:
                    best_mid = abs(yp[j] - 0.5)
                    mid_guess = titr_values[j]

            if best_mid >= 0.485:
                if yp[0] > 0.985:
                    if '+' in res_key:
                        fp.write("%s        >%-24.1f\n" % (res_key, titr_values[-1]))
                    else:
                        fp.write("%s        <%-24.1f\n" % (res_key, titr_values[0]))
                elif yp[-1] > 0.985:
                    if '+' in res_key:
                        fp.write("%s        <%-24.1f\n" % (res_key, titr_values[0]))
                    else:
                        fp.write("%s        >%-24.1f\n" % (res_key, titr_values[-1]))
                else:
                    if abs(yp[0] - yp[-1]) > 0.5:
                        fp.write("%-10s        %-25s\n" % (res_key, "titration curve too sharp"))
                    else:
                        if '+' in res_key:
                            if yp[0] > yp[-1]:
                                fp.write("%s        >%-24.1f\n" % (res_key, titr_values[-1]))
                            else:
                                fp.write("%s        <%-24.1f\n" % (res_key, titr_values[0]))
                        else:
                            if yp[0] < yp[-1]:
                                fp.write("%s        >%-24.1f\n" % (res_key, titr_values[-1]))
                            else:
                                fp.write("%s        <%-24.1f\n" % (res_key, titr_values[0]))
                continue

            result = minimize(
                score, [0.0, mid_guess],
                args=(titr_values, yp),
                method='Nelder-Mead',
                options={'xatol': 0.0001, 'fatol': 0.0001, 'maxiter': 5000},
            )
            a_fit, b_fit = result.x
            chi2 = result.fun

            if titr_type == "ph":
                n_slope = abs(a_fit * 8.617342e-2 * 298.15 / 58.0)
            else:
                n_slope = abs(a_fit * 8.617342e-2 * 298.15)

            if b_fit < titr_values[0]:
                fp.write("%s        <%-24.1f\n" % (res_key, titr_values[0]))
            elif b_fit > titr_values[-1]:
                fp.write("%s        >%-24.1f\n" % (res_key, titr_values[-1]))
            else:
                fp.write("%s    %9.3f %9.3f %9.3f\n" % (res_key, b_fit, n_slope, 1000 * chi2))


def run_parallel(args):
    """Run step 4 in parallel, one mcce process per titration point."""
    base_dir = os.getcwd()
    mcce_exe = os.path.abspath(args.e) if os.path.exists(args.e) else args.e
    runprm_base = build_runprm_dict(args)

    titr_type = runprm_base["TITR_TYPE"]
    initial = float(args.i)
    interval = float(args.d)
    n_steps = int(args.n)
    titr_values = [initial + k * interval for k in range(n_steps)]

    print("   Running step 4 in parallel with %d processors across %d titration points..." % (args.p, n_steps))
    print("   %s points: %s" % (titr_type.upper(), ", ".join("%.1f" % v for v in titr_values)))

    temp_dirs = []
    runprm_list = []
    for k, val in enumerate(titr_values):
        temp_dir = os.path.join(base_dir, "_step4_%s_%.1f" % (titr_type, val))
        temp_dirs.append(temp_dir)
        runprm = dict(runprm_base)
        runprm["TITR_STEPS"] = 1
        if titr_type == "ph":
            runprm["TITR_PH0"] = val
        else:
            runprm["TITR_EH0"] = val
        runprm_list.append(runprm)

    record_runprm(runprm_base, "#STEP4")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.p) as executor:
        futures = {}
        for k in range(n_steps):
            future = executor.submit(
                run_single_point, mcce_exe, base_dir, temp_dirs[k], runprm_list[k]
            )
            futures[future] = titr_values[k]

        for future in as_completed(futures):
            val = futures[future]
            result = future.result()
            completed += 1
            if result:
                print("   [%d/%d] %s=%.1f completed" % (completed, n_steps, titr_type.upper(), val))
            else:
                print("   [%d/%d] %s=%.1f FAILED" % (completed, n_steps, titr_type.upper(), val))

    print("   Merging results...")
    merge_fort38(temp_dirs, titr_type, titr_values)
    merge_mc_out(temp_dirs, titr_type, titr_values)

    print("   Generating pK.out from merged fort.38...")
    generate_pK_out(titr_type)

    for tdir in temp_dirs:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)
    print("   Cleaned up temporary directories.")


def make_pk_extended_and_trim(pk_file="pK.out", pk_extended="pK_extended.out"):
    """
    1) Copy pK.out -> pK_extended.out
    2) Overwrite pK.out keeping only columns up through '1000*chi2'
       (preserves original spacing by slicing at the header's 'vdw0' start).
    """
    if not os.path.isfile(pk_file):
        print(f"WARNING: {pk_file} not found; skipping pK copy/trim.")
        return

    # 1) Archive the full file
    shutil.copy2(pk_file, pk_extended)

    # 2) Determine where to cut based on header
    with open(pk_extended, "r") as fin:
        lines = fin.readlines()

    cut_idx = None
    for line in lines:
        # Find the header line that contains the column names
        if "1000*chi2" in line:
            # Best cut is at the start of the NEXT column (vdw0),
            # so we keep EXACT spacing up to chi2 column.
            if "vdw0" in line:
                cut_idx = line.index("       vdw0")
            else:
                cut_idx = line.index("1000*chi2") + len("1000*chi2")
            break

    if cut_idx is None:
        print("WARNING: Could not find '1000*chi2' header; skipping trim.")
        return

    # Rewrite pK.out with sliced lines (preserving characters/spaces up to cut_idx)
    with open(pk_file, "w") as fout:
        for line in lines:
            if len(line) >= cut_idx:
                fout.write(line[:cut_idx].rstrip("\n") + "\n")
            else:
                # Keep shorter lines as-is (blank lines, etc.)
                fout.write(line)

    print(f"Saved full pKa/Em output as {pk_extended} and trimmed version as {pk_file}")


def sanitycheck_step4(t0):
    """
    Sanity check for step 4. 
    
    Input t0: start time of step 4 run
    Returns:
        STATUS: messages
        STATUS = SUCCESS, WARNING, ERROR
    """
    status = "SUCCESS"
    messages = []

    expected_files = [
                    "mc_out",
                    "fort.38",
                    "pK_extended.out",
                    "sum_crg.out",
                    "pK.out",
    ]

    for fname in expected_files:
        if not os.path.isfile(fname) or os.path.getmtime(fname) < t0:
            status = "ERROR"
            messages.append(f" Output {fname} not generated by this run")

    # Final report
    if status == "SUCCESS":
        messages.append("Step 4 completed successfully.")
    else:
        messages.append(" Check the log to see what went wrong.")
    msg_str = f"[{status}]: " + ";".join(messages)

    return msg_str


if __name__ == "__main__":
    t0 = time.time()

    # Get the command arguments
    helpmsg = "Run mcce step 4, Monte Carlo sampling to simulate a titration."
    parser = argparse.ArgumentParser(description=helpmsg)

    parser.add_argument(
        "--norun",
        default=False,
        action="store_true",
        help="Create run.prm but do not run step 4; default: %(default)s.",
    )
    parser.add_argument(
        "-i",
        metavar="initial ph/eh",
        type=float,
        default="0.0",
        help="Initial pH/Eh of titration; default: %(default)s.",
    )
    parser.add_argument(
        "-d",
        metavar="interval",
        type=float,
        default="1.0",
        help="titration interval in pJ or mV; default: %(default)s.",
    )
    parser.add_argument(
        "-n",
        metavar="steps",
        type=int,
        default="15",
        help="number of steps of titration; default: %(default)s.",
    )
    parser.add_argument(
        "--xts",
        default=False,
        action="store_true",
        help="Enable entropy correction, default is false; default: %(default)s.",
    )
    parser.add_argument(
        "--ms",
        default=False,
        action="store_true",
        help="Enable microstate output; default: %(default)s.",
    )
    parser.add_argument(
        "-e",
        metavar="/path/to/mcce",
        type=str,
        default="mcce",
        help="mcce executable location; default: %(default)s.",
    )
    parser.add_argument(
        "-t",
        metavar="ph or eh",
        type=str,
        default="ph",
        help="titration type, pH or Eh; default: %(default)s.",
    )
    parser.add_argument(
        "-u",
        metavar="Key=Value",
        type=str,
        default="",
        help="User customized variables; default: %(default)s.",
    )
    parser.add_argument(
        "-load_runprm",
        metavar="prm_file",
        default="",
        help="Load additional run.prm file, overwrite default values.",
    )
    parser.add_argument(
        "-p",
        metavar="nproc",
        type=int,
        default=1,
        help="Number of processors for parallel titration; default: %(default)s.",
    )

    args = parser.parse_args()

    detected = detect_runprm()

    if args.p > 1 and not args.norun:
        write_runprm(args)
        run_parallel(args)
    else:
        write_runprm(args)
        if not args.norun:
            process = subprocess.Popen([args.e], stdout=subprocess.PIPE)
            for line in process.stdout:
                print(line.decode(), end="")

    if detected:
        restore_runprm()

    # Amend sum_crg.out, use head3.lst and fort.38 to compose sum_crg.out
    amend_sum_crg()

    # Archive full pK and then trim pK.out to chi2 column
    make_pk_extended_and_trim()

    # Sanity check
    sanity = sanitycheck_step4(t0)
    print(sanity)    