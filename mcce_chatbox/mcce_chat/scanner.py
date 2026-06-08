"""Scan an MCCE4 run directory and build a structured context summary."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunParams:
    inpdb: str = ""
    titr_type: str = ""
    ph_initial: float = 0.0
    ph_final: float = 0.0
    ph_interval: float = 0.0
    eh_initial: float = 0.0
    eh_final: float = 0.0
    eh_interval: float = 0.0
    epsilon_prot: float = 0.0
    do_premcce: bool = False
    do_rotamers: bool = False
    do_energy: bool = False
    do_monte: bool = False
    pbe_solver: str = ""
    extra_params: dict = field(default_factory=dict)


@dataclass
class ProteinInfo:
    pdb_id: str = ""
    chains: list = field(default_factory=list)
    residue_count: int = 0
    hetatm_ligands: list = field(default_factory=list)
    atom_count: int = 0


@dataclass
class ConformerSummary:
    total_conformers: int = 0
    titratable_residues: list = field(default_factory=list)
    residue_conformer_counts: dict = field(default_factory=dict)


@dataclass
class RunContext:
    run_dir: str = ""
    params: Optional[RunParams] = None
    protein: Optional[ProteinInfo] = None
    conformers: Optional[ConformerSummary] = None
    log_tail: list = field(default_factory=list)
    log_errors: list = field(default_factory=list)
    log_warnings: list = field(default_factory=list)
    book_state: dict = field(default_factory=dict)
    steps_done: list = field(default_factory=list)

    def mode(self) -> str:
        if not self.params and not self.protein:
            return "empty"
        states = set(self.book_state.values())
        if "e" in states:
            return "failed"
        if "r" in states:
            return "running"
        if self.steps_done and len(self.steps_done) == 4:
            return "done"
        if self.log_errors:
            return "failed"
        if self.steps_done:
            return "running" if "r" in states else "setup"
        return "setup"

    def summary(self) -> str:
        parts = []
        m = self.mode()
        parts.append(f"Run directory: {self.run_dir}")
        parts.append(f"Run mode: {m}")

        if self.params:
            p = self.params
            parts.append(f"Input PDB: {p.inpdb}")
            if p.titr_type:
                parts.append(f"Titration type: {p.titr_type}")
            if p.titr_type == "ph":
                parts.append(f"pH range: {p.ph_initial} to {p.ph_final} (step {p.ph_interval})")
            elif p.titr_type == "eh":
                parts.append(f"Eh range: {p.eh_initial} to {p.eh_final} (step {p.eh_interval})")
            if p.epsilon_prot:
                parts.append(f"Protein dielectric: {p.epsilon_prot}")
            steps_planned = []
            if p.do_premcce:
                steps_planned.append("1-prerun")
            if p.do_rotamers:
                steps_planned.append("2-rotamers")
            if p.do_energy:
                steps_planned.append("3-energy")
            if p.do_monte:
                steps_planned.append("4-monte")
            if steps_planned:
                parts.append(f"Steps planned: {', '.join(steps_planned)}")
            if p.pbe_solver:
                parts.append(f"PBE solver: {p.pbe_solver}")

        if self.protein:
            pr = self.protein
            if pr.pdb_id:
                parts.append(f"PDB ID: {pr.pdb_id}")
            parts.append(f"Chains: {', '.join(pr.chains) if pr.chains else 'unknown'}")
            parts.append(f"Residues: {pr.residue_count}, Atoms: {pr.atom_count}")
            if pr.hetatm_ligands:
                parts.append(f"Ligands (HETATM): {', '.join(sorted(set(pr.hetatm_ligands)))}")

        if self.conformers:
            c = self.conformers
            parts.append(f"Total conformers: {c.total_conformers}")
            if c.titratable_residues:
                parts.append(f"Titratable residues: {', '.join(c.titratable_residues[:20])}")
                if len(c.titratable_residues) > 20:
                    parts.append(f"  ... and {len(c.titratable_residues) - 20} more")

        if self.steps_done:
            parts.append(f"Steps completed: {', '.join(str(s) for s in sorted(self.steps_done))}")

        if self.book_state:
            summary_counts = {}
            for state in self.book_state.values():
                label = {"i": "idle", "r": "running", "c": "complete", "e": "error"}.get(state, state)
                summary_counts[label] = summary_counts.get(label, 0) + 1
            parts.append(f"Job states: {', '.join(f'{v} {k}' for k, v in summary_counts.items())}")

        if self.log_errors:
            parts.append(f"Errors in log ({len(self.log_errors)}):")
            for e in self.log_errors[:5]:
                parts.append(f"  {e.strip()}")

        if self.log_warnings:
            parts.append(f"Warnings in log: {len(self.log_warnings)}")

        return "\n".join(parts)


_PRM_KEY_MAP = {
    "(INPDB)": "inpdb",
    "(TIESSION)": "titr_type",
    "(TITR_TYPE)": "titr_type",
    "(PH_I)": "ph_initial",
    "(PH_F)": "ph_final",
    "(PH_INT)": "ph_interval",
    "(EH_I)": "eh_initial",
    "(EH_F)": "eh_final",
    "(EH_INT)": "eh_interval",
    "(EPSILON_PROT)": "epsilon_prot",
    "(DO_PREMCCE)": "do_premcce",
    "(DO_ROTAMERS)": "do_rotamers",
    "(DO_ENERGY)": "do_energy",
    "(DO_MONTE)": "do_monte",
    "(PBE_SOLVER)": "pbe_solver",
}

_BOOL_KEYS = {"do_premcce", "do_rotamers", "do_energy", "do_monte"}
_FLOAT_KEYS = {"ph_initial", "ph_final", "ph_interval", "eh_initial", "eh_final",
                "eh_interval", "epsilon_prot"}

_TITRATABLE = {
    "ASP", "GLU", "HIS", "LYS", "ARG", "TYR", "CYS",
    "CTR", "NTR", "NTG",
    "HEM", "PAA", "PDD",
}


def parse_run_prm(path: str) -> RunParams:
    params = RunParams()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.search(r"\(([A-Z_]+)\)", line)
                if not m:
                    continue
                key_tag = f"({m.group(1)})"
                value = line[:m.start()].strip().split("#")[0].strip()
                if not value:
                    continue
                value = value.split()[0]
                attr = _PRM_KEY_MAP.get(key_tag)
                if attr:
                    if attr in _BOOL_KEYS:
                        setattr(params, attr, value.lower() in ("t", "true", "1", "yes"))
                    elif attr in _FLOAT_KEYS:
                        try:
                            setattr(params, attr, float(value))
                        except ValueError:
                            pass
                    else:
                        setattr(params, attr, value)
                else:
                    params.extra_params[m.group(1)] = value
    except (OSError, IOError):
        return None
    return params


def parse_step1_pdb(path: str) -> ProteinInfo:
    info = ProteinInfo()
    chains = set()
    residues = set()
    hetatms = set()
    atom_count = 0
    try:
        with open(path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM"):
                    atom_count += 1
                    chain = line[21] if len(line) > 21 else ""
                    resname = line[17:20].strip() if len(line) > 19 else ""
                    resseq = line[22:26].strip() if len(line) > 25 else ""
                    if chain and chain != " ":
                        chains.add(chain)
                    if resname and resseq:
                        residues.add((chain, resname, resseq))
                    if rec == "HETATM" and resname not in ("HOH", "WAT", "DOD", "TIP"):
                        hetatms.add(resname)
                elif rec == "HEADER" and len(line) > 62:
                    pdb_id = line[62:66].strip()
                    if pdb_id:
                        info.pdb_id = pdb_id
    except (OSError, IOError):
        return None

    info.chains = sorted(chains)
    info.residue_count = len(residues)
    info.hetatm_ligands = sorted(hetatms)
    info.atom_count = atom_count
    return info


def parse_head3_lst(path: str) -> ConformerSummary:
    summary = ConformerSummary()
    titr_set = set()
    counts = {}
    try:
        with open(path) as f:
            header = f.readline()
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                conf_name = parts[1] if len(parts) > 1 else parts[0]
                summary.total_conformers += 1
                res3 = conf_name[:3] if len(conf_name) >= 3 else conf_name
                res_id = conf_name[:3] + conf_name[5:11] if len(conf_name) >= 11 else conf_name
                counts[res_id] = counts.get(res_id, 0) + 1
                if res3 in _TITRATABLE:
                    titr_set.add(res_id)
    except (OSError, IOError):
        return None

    summary.titratable_residues = sorted(titr_set)
    summary.residue_conformer_counts = counts
    return summary


def parse_run_log(path: str, tail_lines: int = 30) -> tuple:
    lines = []
    errors = []
    warnings = []
    try:
        with open(path) as f:
            all_lines = f.readlines()
        lines = all_lines[-tail_lines:]
        for l in all_lines:
            ll = l.lower()
            if "error" in ll or "fatal" in ll or "traceback" in ll:
                errors.append(l.rstrip())
            elif "warning" in ll or "warn" in ll:
                warnings.append(l.rstrip())
    except (OSError, IOError):
        pass
    return lines, errors, warnings


def parse_book_txt(path: str) -> dict:
    state = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    state[parts[0]] = parts[-1]
    except (OSError, IOError):
        pass
    return state


def detect_steps_done(run_dir: str) -> list:
    done = []
    sentinels = {
        1: ["step1_out.pdb"],
        2: ["head3.lst"],
        3: ["energies"],
        4: ["pK.out", "fort.38"],
    }
    for step, files in sentinels.items():
        for f in files:
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                if os.path.isdir(p):
                    if os.listdir(p):
                        done.append(step)
                        break
                else:
                    done.append(step)
                    break
    return done


def scan_run_directory(run_dir: str) -> RunContext:
    run_dir = os.path.abspath(run_dir)
    ctx = RunContext(run_dir=run_dir)

    prm_path = os.path.join(run_dir, "run.prm")
    if os.path.isfile(prm_path):
        ctx.params = parse_run_prm(prm_path)

    pdb_path = os.path.join(run_dir, "step1_out.pdb")
    if os.path.isfile(pdb_path):
        ctx.protein = parse_step1_pdb(pdb_path)

    head3_path = os.path.join(run_dir, "head3.lst")
    if os.path.isfile(head3_path):
        ctx.conformers = parse_head3_lst(head3_path)

    log_path = os.path.join(run_dir, "run.log")
    if os.path.isfile(log_path):
        ctx.log_tail, ctx.log_errors, ctx.log_warnings = parse_run_log(log_path)

    book_path = os.path.join(run_dir, "book.txt")
    if os.path.isfile(book_path):
        ctx.book_state = parse_book_txt(book_path)

    ctx.steps_done = detect_steps_done(run_dir)

    return ctx
