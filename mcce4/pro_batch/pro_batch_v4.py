#!/usr/bin/env python3

"""
Module: probatch.py

Codebase of the pro_batch tool

NOTES:
1. The current implementation creates a submit script in the current working
   directory (cwd), and considers it as the master: this is where the users 
   would enter their modifications.
   The user's modifications will be used for each protein simulation as this
   master script file is soft-linked into each protein folder.
"""
import argparse
import json
import os
from pathlib import fnmatch, Path
from re import split as re_split
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Tuple, Union

from mcce4 import CLI_EPILOG, CLONE, CLONE_PATH 
from mcce4.downloads import get_rcsb_pdb
from mcce4.io_utils import subprocess_run, CalledProcessError
from mcce4.protinfo.cli import get_pdb_rpt, prerun_passed


datasets_dict = None
inputpath_help ="""
Path of folder containing PDB files
OR Path of file listing pdb filepaths or pdbids, which will be downloaded
"""

if CLONE == "MCCE4":
    # Attempt to import MCCE4 benchmark resources if available
    try:
        from mcce4.mcce_benchmark import BenchResources, datasets_dict
        inputpath_help += f"OR Name of a pkdb dataset, one of: {list(datasets_dict.keys())}."
    except ImportError:
        pass


SUBMIT_SCRIPT = "submit_mcce4.sh"
SUBMIT_SCRIPT_PATH = CLONE_PATH.joinpath("schedulers", SUBMIT_SCRIPT)
DEFAULT_JOBNAME = "mcce_batch"
DEFAULT_MAX_JOBS = 15
DEFAULT_NICE = 10
DEFAULT_SLURM_NICE = 1000
CUSTOM_PRM = "run.prm.custom"
BAD_ANSWER = "Answer must be one of [yes, no, y, n] (case insensitive). Please try again."

# Related to the creation/update of the bookkeeping file:
BOOK = "book.txt"
N_HEADER_LINES = 3
DELIM_LINE = "-" * 38 + "\n"
BOOK_FOOTER_LINES = [
    DELIM_LINE,
    "Legend:\n",
    " r : Pending, Ready or Running\n",
    " c : Completed (pK.out found)\n",
    " e : Error (Check run.log in directory or bench_book.txt for Failed @ prerun)\n"
]
N_FOOTER_LINES = len(BOOK_FOOTER_LINES)

JOBS_FILE = ".pro_batch_jobs.json"
JOBS_FP = Path(JOBS_FILE)
SQUEUE_FMT = "%.10i %.9P %.30j %.8u %.2t %.10M %.10L %.6D %R"
_log_fh = None


def _load_tracking() -> dict:
    if JOBS_FP.exists():
        return json.loads(JOBS_FP.read_text())
    return {}


def _save_tracking(data: dict):
    if data:
        JOBS_FP.write_text(json.dumps(data, indent=2))
    elif JOBS_FP.exists():
        JOBS_FP.unlink()

    return


class TeeWriter:
    """Duplicates writes to both the terminal and a log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


class JobPool:
    """Manages a pool of concurrent subprocess jobs with a concurrency limit (bash mode)."""

    def __init__(self,
                 max_jobs: int = DEFAULT_MAX_JOBS, nice: int = DEFAULT_NICE,
                 job_name: str = DEFAULT_JOBNAME, poll_interval: float = 10.0):
        self.max_jobs = max_jobs
        self.nice = nice
        self.job_name = job_name
        self.poll_interval = poll_interval
        self.running = {}   # {protein_name: subprocess.Popen}
        self.pids = {}      # {protein_name: pid} for tracking/stop

    def _poll(self):
        """Check running jobs, remove finished ones."""
        finished = []
        for name, proc in self.running.items():
            retcode = proc.poll()
            if retcode is not None:
                finished.append(name)
                if retcode != 0:
                    print(f"  !! {name} exited with code {retcode} — check run.log")

        for name in finished:
            del self.running[name]

    def _wait_for_slot(self):
        """Block until at least one job slot is available."""
        while len(self.running) >= self.max_jobs:
            self._poll()
            if len(self.running) >= self.max_jobs:
                time.sleep(self.poll_interval)

    def launch(self, protein_name: str, p_dir: Path, script_name: str):
        """Launch a single protein run inside p_dir."""
        self._wait_for_slot()

        cmd = ["nice", "-n", str(self.nice), "bash", script_name]
        log_fh = open(p_dir / "run.log", "w")

        proc = subprocess.Popen(
            cmd,
            cwd=str(p_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )
        log_fh.close()
        self.running[protein_name] = proc
        self.pids[protein_name] = proc.pid
        print(f"  -> Started {protein_name} (PID {proc.pid}, "
              f"nice {self.nice}, {len(self.running)}/{self.max_jobs} slots)")
        self.save_tracking()

    def save_tracking(self):
        data = _load_tracking()
        data[self.job_name] = {
            "mode": "bash",
            "jobs": self.pids,
            "launched_at": datetime.now().isoformat(),
        }
        _save_tracking(data)

    def summary(self):
        print(f"\n{'=' * 45}")
        print(f"Job Pool Summary: {len(self.pids)} jobs launched in background")
        print(f"  Job name      : {self.job_name}")
        print(f"  Nice priority : {self.nice}")
        print(f"  Max concurrent: {self.max_jobs}")
        print(f"  Check status  : pro_batch <input_path> --check -job_name {self.job_name}")
        print(f"  Stop batch    : pro_batch --stop -job_name {self.job_name}")
        print(f"{'=' * 45}")


class SlurmPool:
    """Manages Slurm job submissions with a concurrency limit by polling squeue."""

    def __init__(self,
                 max_jobs: int = DEFAULT_MAX_JOBS, nice: int = DEFAULT_SLURM_NICE,
                 job_name: str = DEFAULT_JOBNAME, poll_interval: float = 30.0):
        self.max_jobs = max_jobs
        self.nice = nice
        self.job_name = job_name
        self.poll_interval = poll_interval
        self.job_ids = {}    # {protein_name: slurm_job_id}

    def _count_running(self) -> int:
        """Count our active (running + pending) Slurm jobs by job name."""
        try:
            result = subprocess.run(
                ["squeue", "--me", "--name", self.job_name,
                 "--noheader", "--states", "R,PD"],
                capture_output=True, text=True, timeout=15
            )
            lines = result.stdout.strip()
            return len(lines.splitlines()) if lines else 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Warning: squeue check failed ({e}), assuming 0 running")
            return 0

    def _wait_for_slot(self):
        """Block until Slurm has fewer than max_jobs running/pending."""
        while True:
            running = self._count_running()
            if running < self.max_jobs:
                return
            print(f"  ... {running}/{self.max_jobs} Slurm slots occupied, "
                  f"waiting {self.poll_interval}s...")
            time.sleep(self.poll_interval)

    def launch(self, protein_name: str, p_dir: Path, script_name: str):
        """Submit a single protein run via sbatch with nice priority and job name."""
        self._wait_for_slot()

        cmd = ["sbatch", f"--nice={self.nice}",
               f"--job-name={self.job_name}", script_name]
        result = subprocess.run(cmd, cwd=str(p_dir), capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "?"
            self.job_ids[protein_name] = job_id
            print(f"  -> Submitted {protein_name} (Slurm job {job_id}, "
                  f"nice {self.nice}, {len(self.job_ids)} total)")
            self.save_tracking()
        else:
            print(f"  !! sbatch failed for {protein_name}: {result.stderr.strip()}")

    def save_tracking(self):
        data = _load_tracking()
        data[self.job_name] = {
            "mode": "slurm",
            "jobs": self.job_ids,
            "launched_at": datetime.now().isoformat(),
        }
        _save_tracking(data)

    def summary(self):
        print(f"\n{'=' * 45}")
        print(f"Slurm Summary: {len(self.job_ids)} jobs submitted")
        print(f"  Job name      : {self.job_name}")
        print(f"  Nice priority : {self.nice}")
        print(f"  Max concurrent: {self.max_jobs}")
        print(f"  Slurm status  : squeue --me --name {self.job_name} -o \"{SQUEUE_FMT}\"")
        print(f"  Check status  : pro_batch <input_path> --check -job_name {self.job_name}")
        print(f"  Stop batch    : pro_batch --stop -job_name {self.job_name}")
        print(f"{'=' * 45}")


def _stop_batch(batch_name: str, batch: dict):
    """Stop a single tracked batch."""
    mode = batch["mode"]
    jobs = batch.get("jobs", {})
    launched = batch.get("launched_at", "unknown")

    print(f"\n  Batch '{batch_name}' ({len(jobs)} jobs, mode: {mode}, launched: {launched})")

    if not jobs:
        print("    No jobs in this batch.")
        return

    if mode == "slurm":
        job_ids = [str(jid) for jid in jobs.values() if str(jid) != "?"]
        if job_ids:
            result = subprocess.run(["scancel"] + job_ids,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    Cancelled {len(job_ids)} Slurm jobs: {', '.join(job_ids)}")
            else:
                stderr = result.stderr.strip()
                if stderr:
                    print(f"    scancel output: {stderr}")
                print(f"    Sent cancel for {len(job_ids)} jobs "
                      "(some may have already finished)")
        else:
            print("    No valid Slurm job IDs found.")
    else:
        killed = 0
        already_done = 0
        for name, pid in jobs.items():
            try:
                os.killpg(int(pid), signal.SIGTERM)
                killed += 1
                print(f"    Stopped {name} (PID {pid})")
            except ProcessLookupError:
                already_done += 1
            except PermissionError:
                print(f"    {name} (PID {pid}): permission denied")
        print(f"    Stopped {killed}, already finished {already_done}.")


def stop_jobs(job_name: str = None):
    """Stop tracked jobs for a specific batch, or all batches if job_name is None."""
    data = _load_tracking()
    if not data:
        print("No tracked jobs found. Nothing to stop.")
        return

    if job_name:
        if job_name not in data:
            print(f"No tracked batch named '{job_name}'.")
            print(f"  Tracked batches: {', '.join(data.keys())}")
            return
        _stop_batch(job_name, data[job_name])
        del data[job_name]
    else:
        print(f"Stopping all {len(data)} tracked batch(es)...")
        for name in list(data.keys()):
            _stop_batch(name, data[name])
        data.clear()

    _save_tracking(data)
    update_book()
    print("\nBook status updated.")

    return


def _print_squeue(job_name: str):
    """Print squeue output for a given job name."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--name", job_name, "-o", SQUEUE_FMT],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        if output:
            print(f"\nSlurm queue for '{job_name}':", output, sep="\n")
        else:
            print(f"\nNo active Slurm jobs for '{job_name}'.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return


def check_jobs(job_name: str = None):
    """Print status for a specific batch's proteins, or all if job_name is None."""
    data = _load_tracking()
    update_book()

    if job_name:
        if not data or job_name not in data:
            print(f"No tracked batch named '{job_name}'.")
            if data:
                print(f"  Tracked batches: {', '.join(data.keys())}")
            return
        batch = data[job_name]
        proteins = set(batch.get("jobs", {}).keys())
        mode = batch.get("mode", "?")
        launched = batch.get("launched_at", "?")

        print(f"Batch '{job_name}' (mode: {mode}, launched: {launched})")
        print(f"{'PDB':<12} {'Status':<8} {'Last_Step'}")
        print(DELIM_LINE.strip())
        for line in Path(BOOK).read_text().splitlines()[N_HEADER_LINES:-N_FOOTER_LINES]:
            parts = line.split()
            if parts and parts[0].upper() in proteins:
                print(line)

        if mode == "slurm":
            _print_squeue(job_name)
    else:
        print("Book status refreshed in", BOOK)
        if data:
            print(f"\nTracked batches:")
            for name, batch in data.items():
                n = len(batch.get("jobs", {}))
                mode = batch.get("mode", "?")
                launched = batch.get("launched_at", "?")
                print(f"  {name:<20} {n} jobs, mode: {mode}, launched: {launched}")
                if mode == "slurm":
                    _print_squeue(name)


# FIX get_protein_status:
#  If submit script is running only steps 1 & 2, the updated book will show:
#    <pdbid> r        In Step3
#  even though the run is over (no running job on that pdbid).
#  Instead, it would be preferable to show:
#    <pdbid> c        Completed Step2
#
def get_protein_status(protein_name: str) -> Tuple[str, str]:
    """Determines status based on file presence in the protein directory."""
    if protein_name.startswith("#"):
        return "e", "Failed @ prerun"

    p_dir = Path(protein_name.upper())
    if not p_dir.exists():
        return "r", "Pending"

    if not (p_dir / "run.log").exists():
        return "r", "Pending"

    steps = [
        ("step1_out.pdb", "Step1"),
        ("step2_out.pdb", "Step2"),
        ("head3.lst",     "Step3"),
        ("pK.out",        "Step4"),
    ]

    last_valid_step = "None"
    for filename, step_label in steps:
        if (p_dir / filename).exists():
            last_valid_step = step_label
        else:
            break

    if last_valid_step == "Step4":
        return "c", "Completed"

    if last_valid_step == "None":
        return "e", "Failed @ Step1"

    next_idx = int(last_valid_step[-1]) + 1
    return "r", f"In Step{next_idx}"


def book_header_lines() -> list:
    return [f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"{'PDB':<12} {'Status':<8} {'Last_Step'}\n",
            DELIM_LINE]


def write_book(user_files: list, bench_book: bool = False):
    """Write the bookkeeping file with pdb ids from the user files
    including a clean status table, timestamp, and legend.
    If user_files is empty and bench_book is True, use pre-existing book.txt,
    (presumably created by bench_setup).
    """
    # preset with header lines:
    content = book_header_lines()
    # convert Path to str for sorted:
    user_files = [str(fp) if isinstance(fp, Path) else fp[0] for fp in user_files]
    if user_files:
        for entry in sorted(user_files):
            if isinstance(entry, tuple):
                content.append(f"{entry[0]:<12} {'r':<8} Pending\n")
            else:
                # entry was a Path
                content.append(f"{Path(entry).stem.upper():<12} {'r':<8} Pending\n")
    else:
        if not bench_book:
            print("ERROR: With empty user_files, bench_book must be True (and book.txt exist).")
            sys.exit(1)

        book_matches = sorted(Path.cwd().glob('*book.txt'))
        if not book_matches:
            print("ERROR: No *book.txt file found in current directory.")
            sys.exit(1)
        bench_book_fp = book_matches[0]

        previous_book = f"bench_{BOOK}"
        if bench_book_fp.name != previous_book:
            shutil.copy2(bench_book_fp, Path(previous_book))

        for line in bench_book_fp.read_text().splitlines():
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if not line.startswith("#"):
                content.append(f"{parts[0]:<12} {'r':<8} Pending\n")
            elif len(parts) >= 3:
                content.append(f"{parts[0]:<12} {'e':<8} {parts[2][2:]}\n")

    content.extend(BOOK_FOOTER_LINES)
    with open(BOOK, "w") as f:
        f.writelines(content)


def update_book():
    """Updates book.txt with a clean status table, timestamp, and legend."""
    book_fp = Path(BOOK)
    if not book_fp.exists():
        return

    protein_ids = []
    for line in book_fp.read_text().splitlines()[N_HEADER_LINES:-N_FOOTER_LINES]:
        parts = line.split()
        if parts:
            protein_ids.append(parts[0].upper())

    # preset with header lines
    content = book_header_lines()
    for pid in protein_ids:
        flag, step = get_protein_status(pid)
        content.append(f"{pid:<12} {flag:<8} {step}\n")
    content.extend(BOOK_FOOTER_LINES)

    with open(BOOK, "w") as f:
        f.writelines(content)


def comment_book_pdb(pdb_dir: Path, tag: str):
    """Comment out & flag pdbid line in book.txt with error status ('e'), 
    followed by a tag (reason) string.
    """
    pdbid = pdb_dir.name
    book_fp = pdb_dir.parent.joinpath(BOOK)
    if not book_fp.exists():
        print("book.txt not found for commenting!")
        return
    cmd = f"sed -i 's/^\({pdbid}*\).*$/#{pdbid}        e\t{tag}/' {book_fp!s}"
    subprocess_run(cmd)

    return


def modify_script_for_runprm(script_path: Path):
    """Injects custom run.prm loading into the shell script steps."""
    if not Path(CUSTOM_PRM).exists():
        return
    load_custom_str = f" -load_runprm {CUSTOM_PRM}\n"
    new_lines = []
    for line in script_path.read_text().splitlines():
        parts = line.split()
        if parts and fnmatch.fnmatch(parts[0], "step[1234].py"):
            new_lines.append(line.strip() + load_custom_str)
        else:
            new_lines.append(line + "\n")
    with open(script_path, "w") as f:
        f.writelines(new_lines)


def _ensure_symlink(link_path: Path, target: str):
    """Create or replace a symlink at link_path pointing to target."""
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and os.readlink(str(link_path)) == target:
            return
        if not link_path.is_symlink():
            link_path.unlink()
    link_path.symlink_to(target)

    return


def _get_user_files_from_file(input_fp: Path) -> Tuple[List[Union[Path, str]], bool]:
    """Case where input_fp is a file that contains pdbs file paths or pdbids.
    """
    pdbs_lst = []
    with open(input_fp) as fh:
        for lin in fh:
            line = lin.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            # allow for multiple fields, space or comma separated:
            parts = re_split(r"[ ,]+", line)
            pdb_fp = Path(parts[0])
            if pdb_fp.is_file():
                # file: validate & add to list; ignore other text on line
                if pdb_fp.is_symlink():
                    print(f"ERROR: Cannot use a linked file as pdb source: {pdb_fp!s}")
                    continue
                if pdb_fp.suffix != ".pdb":
                    print(f"ERROR: Cannot use a non-pdb file: {pdb_fp!s}")
                    continue
                pdbs_lst.append(pdb_fp)
            else:
                # non-existent file -> pdbid
                # assumed format:  pdb_id assembly_id [whatever, usually res count]
                bid = 1  # default bioassembly id
                if len(parts) > 1:
                    if parts[1].isnumeric():
                        if int(parts[1]) > 0:
                            bid = int(parts[1])
                pdbs_lst.append((pdb_fp.stem, bid))  # download bioassemb

        return pdbs_lst, Path(BOOK).exists()


def _get_user_files_from_dir(input_dir: Path) -> Tuple[List[Path], bool]:
    """Return a 2-tuple:
        (list of .pdb files that are neither mcce outputs nor prot.pdb, [False, True])
    The second Boolean item indicates whether a pre-existing book.txt file was found.
    """
    return [fp for fp in input_dir.glob("*.pdb")
            if not (fnmatch.fnmatch(fp.name, "step[0123]_out.pdb")
                    or fp.name == "prot.pdb")], Path(BOOK).exists()


def get_user_files(input_path: str) -> Tuple[List[Path], bool]:
    """Semaphore function to obtain list of user pdb filepaths from
    a directory or from a file (containing pdbs paths or pdbids).
    """
    input_fp = Path(input_path)
    if not input_fp.exists():
        print(f"ERROR: {input_path} does not exist.")
        sys.exit(1)

    if input_fp.is_dir():
        return _get_user_files_from_dir(input_fp)
    else:
        return _get_user_files_from_file(input_fp)


def ask_user(question: str) -> bool:
    while True:
        res = input(f"{question} (y/n): ").lower().strip()
        if _log_fh:
            _log_fh.write(res + "\n")
            _log_fh.flush()
        if not res:
            print(BAD_ANSWER)
            continue
        if res[0] == "y":
            return True
        if res[0] == "n":
            return False
        print(BAD_ANSWER)


def do_prerun(p_dir: Path, pdb_fp: Path, is_pdbid: bool = False) -> Path:
    """Run protinfo in p_dir.
    Return the report path.
    """
    if is_pdbid:
        prot_path = p_dir.name
    else:
        prot_path = Path(pdb_fp.name)

    os.chdir(p_dir)

    fetch = True if is_pdbid else False
    info_args = {
        "pdb": prot_path,
        "d": 8,
        "u": "",
        "wet": False,
        "noter": False,
        "fetch": fetch,
        "save_dicts": True,
    }
    # run step1 and write prot report (from run1.log):
    # get_pdb_rpt creates the 'prerun' output subfolder in the current run:
    rpt_pdb = get_pdb_rpt(argparse.Namespace(**info_args), do_checks=True, do_fetch=fetch)
    os.chdir(Path.cwd().parent)

    return rpt_pdb


def process_protein_file(protein_path: Union[Path, tuple],
                         script_path: Path,
                         pool: Union["JobPool", "SlurmPool"],
                         dry_run: bool = False):
    """Sets up the protein directory and launches the run via the pool.
    """
    is_pdbid = isinstance(protein_path, tuple)
    if is_pdbid:
        # protein_path is a 2-tuple: pdbid, bioassembly#
        p_dir = Path(protein_path[0].upper())
        pdb_fp = p_dir.joinpath(protein_path[0].lower() + ".pdb")
    else:
        protein_path = Path(protein_path)
        p_dir = Path(protein_path.stem.upper())
        pdb_fp = p_dir.joinpath(protein_path.name)

    p_dir.mkdir(exist_ok=True)

    if not (pdb_fp.exists() or is_pdbid):
        shutil.copy2(protein_path, p_dir)
    
    rpt_pdb = do_prerun(p_dir, pdb_fp, is_pdbid=is_pdbid)
    print(f"Prerun report: {rpt_pdb!s}")
    # prerun_passed -> (bool, reason)
    ok, msg = prerun_passed(p_dir.joinpath("prerun"))
    if not ok:
        print(f"Commenting book for {p_dir.name}: {msg}")
        comment_book_pdb(p_dir, tag=msg)

    _ensure_symlink(p_dir/"prot.pdb", pdb_fp.name)

    if Path(CUSTOM_PRM).exists():
        _ensure_symlink(p_dir/CUSTOM_PRM, str(Path(CUSTOM_PRM).absolute()))

    relative_script = os.path.relpath(str(script_path), str(p_dir))
    _ensure_symlink(p_dir/script_path.name, relative_script)

    if not dry_run:
        pool.launch(pdb_fp.stem.upper(), p_dir, script_path.name)

    return


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pro_batch",
                                     description="Batch launch MCCE runs.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=CLI_EPILOG,
                                     )
    # Note: dashes are not converted for positional options:
    parser.add_argument("input_path",
                        type=str,
                        nargs="?",
                        default=None,
                        help=inputpath_help
                        )
    parser.add_argument("-custom",
                        type=str,
                        default="",
                        help="Path to custom bash script."
                        )
    parser.add_argument("-job-name",
                        type=str,
                        default=None,
                        help="""A unique name for this batch (e.g. 'pH7_mutations').
Required for launching, --check, and --stop. (default: %(default)s)"""
                        )
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=DEFAULT_MAX_JOBS,
                        help="Max concurrent jobs: how many MCCE runs execute simultaneously. (default: %(default)s)"
                        )
    parser.add_argument("--nice",
                        type=int,
                        default=None,
                        help="""Lower priority (higher values): nicer to other users.
  Slurm mode: range 0-10000.
  Bash mode (--no_slurm), higher priority: range 0-19."""
                        )
    parser.add_argument("--no-slurm",
                        action="store_true",
                        default=False,
                        help=f"""Use bash instead of the Slurm scheduler to run jobs.
NOTE: Jobs are still low priority ({DEFAULT_NICE}) unless --nice is set with a lower value.
(default: %(default)s)"""
                        )
    parser.add_argument("--check",
                        action="store_true",
                        default=False,
                        help="Update book.txt and show status for the batch given by -job_name. "
                        "(default: %(default)s)"
                        )
    parser.add_argument("--check-all",
                        action="store_true",
                        default=False,
                        help="Update book.txt and show status for all tracked batches. "
                             "(default: %(default)s)"
                        )
    parser.add_argument("--stop",
                        action="store_true",
                        default=False,
                        help="Stop jobs for the batch given by -job_name. (default: %(default)s)"
                        )
    parser.add_argument("--stop-all",
                        action="store_true",
                        default=False,
                        help="Stop all tracked jobs from every batch in this directory. (default: %(default)s)"
                        )
    parser.add_argument("--dry-run",
                        action="store_true",
                        default=False,
                        help="Do setup & prerun only: no job launch. (default: %(default)s)"
                        )

    return parser


def protein_batch(args: Union[argparse.Namespace, dict]):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)

    if args.stop_all:
        stop_jobs()
        return

    if args.stop:
        if not args.job_name:
            print("ERROR, protein_batch: --stop requires -job-name. Use --stop_all to stop every batch.")
        stop_jobs(job_name=args.job_name)
        return

    if args.check_all:
        check_jobs()
        return

    if args.check:
        if not args.job_name:
            print("ERROR, protein_batch: --check requires -job-name. Use --check-all to see every batch.")
        check_jobs(job_name=args.job_name)
        return

    if not args.input_path:
        print("NOTE, protein_batch: input_path is required when launching jobs.")

    if not args.job_name:
        print("NOTE, protein_batch: -job_name is required when launching jobs.")

    # Data path resolution
    if datasets_dict and args.input_path in datasets_dict:
        input_path = str(BenchResources(args.input_path).BENCH_PDBS)
    else:
        input_path = args.input_path
        if not Path(input_path).exists():
            print(f"ERROR, protein_batch: {input_path} does not exist.")
            sys.exit(1)

    # SETUP: proteins folders
    user_files, found_book = get_user_files(input_path)
    if not user_files and not found_book:
        print("ERROR: Cannot proceed: No valid PDB files found & no pre-existing book.")
        sys.exit(1)

    # SETUP: write initial book
    write_book(user_files, found_book)
    update_book()

    # SETUP: default submit script
    script_path = None
    submit_fp = Path(SUBMIT_SCRIPT)
    if args.custom:
        script_path = Path(args.custom)
    elif submit_fp.exists():
        script_path = submit_fp

    if not script_path:
        if SUBMIT_SCRIPT_PATH.exists():
            shutil.copy2(SUBMIT_SCRIPT_PATH, submit_fp)
            submit_fp.chmod(0o755)
            print(f"Created default {SUBMIT_SCRIPT}. Edit it then re-run.",
                  "The proteins listed in the book file will be setup, then launched.",
                  sep="\n")
            sys.exit(0)
        else:
            print(f"ERROR: Submission script {SUBMIT_SCRIPT_PATH!s} not found.")
            sys.exit(1)

    # SETUP: console logging — all output from here is tee'd to the log file
    global _log_fh
    log_name = f"pro_batch_{args.job_name}.log"
    _log_fh = open(log_name, "a")
    _log_fh.write(f"\n{'=' * 60}\n")
    _log_fh.write(f"pro_batch run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_fh.write(f"Command: {' '.join(sys.argv)}\n")
    _log_fh.write(f"{'=' * 60}\n")
    sys.stdout = TeeWriter(sys.stdout, _log_fh)

    # SETUP: Categorize proteins prior to launch
    error_prots = []
    other_prots = []
    new_prots = []

    for line in Path(BOOK).read_text().splitlines()[N_HEADER_LINES:-N_FOOTER_LINES]:
        parts = line.split()
        if len(parts) < 2 or len(parts[0]) < 3 or parts[0].startswith("-"):
            continue
        pdb_name = parts[0].upper()
        status = parts[1]
        detail = parts[2] if len(parts) > 2 else ""

        if not detail or detail == "Pending":
            new_prots.append(pdb_name)
        elif status == 'e' or pdb_name.startswith("#"):
            error_prots.append(pdb_name)
        else:
            other_prots.append(pdb_name)

    final_targets = []

    if new_prots:
        print(f"\nFound {len(new_prots)} new or missing runs to launch: {', '.join(new_prots)}")
        final_targets.extend(new_prots)

    if error_prots:
        print(f"\nFound {len(error_prots)} existing runs with errors ('e'): {', '.join(error_prots)}")
        if ask_user("Would you like to re-run these error cases?"):
            # uncomment any commented pdbids:
            final_targets.extend([pdb[1:] if pdb.startswith("#") else pdb for pdb in error_prots])

    if other_prots:
        print(f"\nFound {len(other_prots)} existing runs (Completed 'c' or Pending/Ready/Running 'r').")
        if ask_user("Would you like to re-run/overwrite these as well?"):
            final_targets.extend(other_prots)

    if not final_targets:
        print("\nNo proteins selected for processing. Exiting.")
        return

    modify_script_for_runprm(script_path)

    # SETUP: prep launch jobs
    if shutil.which("sbatch") is None and not args.no_slurm:
        if ask_user("Slurm is not installed, use bash for submitting instead?"):
            args.no_slurm = True
        else:
            print("Exiting. To install on Linux, run: sudo apt install slurm")
            sys.exit(0)

    if args.no_slurm:
        nice = args.nice if args.nice is not None else DEFAULT_NICE
        pool = JobPool(max_jobs=args.jobs, nice=nice, job_name=args.job_name)
        mode_str = f"bash (max {args.jobs} concurrent, nice {nice})"
    else:
        nice = args.nice if args.nice is not None else DEFAULT_SLURM_NICE
        pool = SlurmPool(max_jobs=args.jobs, nice=nice, job_name=args.job_name)
        mode_str = f"Slurm (max {args.jobs} concurrent, nice {nice})"

    print(f"\nLaunching {len(final_targets)} runs via {mode_str}...")
    print(f"  Log file: {log_name}")
    print("Backgrounding — you can continue using this terminal.\n")

    pool.save_tracking()
    _log_fh.flush()
    pid = os.fork()

    if pid > 0:
        sys.stdout = sys.stdout.terminal if isinstance(sys.stdout, TeeWriter) else sys.__stdout__
        sys.stderr = sys.stderr.terminal if isinstance(sys.stderr, TeeWriter) else sys.__stderr__
        _log_fh.close()
        return

    # Child: detach from terminal, run jobs in background
    os.setsid()
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 0)
    log_fd = _log_fh.fileno()
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(devnull_fd)
    sys.stdin = open(os.devnull, "r")
    sys.stdout = _log_fh
    sys.stderr = _log_fh

    # SETUP & LAUNCH each protein
    #print(f"{user_files = }")

    target_set = set(final_targets)
    for entry in user_files:
        which  = entry[0].upper() if isinstance(entry, tuple) else entry.stem.upper()
        if which in target_set:
            process_protein_file(entry, script_path, pool, args.dry_run)

    pool.save_tracking()
    update_book()
    pool.summary()
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log_fh.close()
    os._exit(0)


def cli():
    p = cli_parser()
    args = p.parse_args()
    protein_batch(args)


if __name__ == "__main__":
    cli()
