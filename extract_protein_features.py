#!/usr/bin/env python3
"""
Comprehensive protein-level feature extraction from MCCE4 simulation directories.

Extracts structural, chemical, and electrostatic features from MCCE4 output files
and writes one CSV where each row is a PDB and each column is a feature.

Usage:
    python extract_protein_features.py                         # default: current dir, pH 7
    python extract_protein_features.py -d /path/to/pdbs        # custom input directory
    python extract_protein_features.py -o my_features.csv      # custom output path
    python extract_protein_features.py --ph 4.0                # different pH
"""
import argparse
import csv
import math
import os
import sys
import time

from analyze_mcce_proteins import (
    list_pdb_dirs, parse_acc_res, parse_sum_crg, get_charge_at_ph, calc_pi,
    parse_pk_out, _init_dipole, _compute_dipole,
    PKA0, IONIZABLE_RES, ACIDS, BASES, RT_LN10, STANDARD_RES,
)

# ── Constants ────────────────────────────────────────────────────────────────

KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

RESIDUE_MW = {
    "ALA": 89.09, "ARG": 174.20, "ASN": 132.12, "ASP": 133.10,
    "CYS": 121.16, "GLN": 146.15, "GLU": 147.13, "GLY": 75.03,
    "HIS": 155.16, "ILE": 131.17, "LEU": 131.17, "LYS": 146.19,
    "MET": 149.21, "PHE": 165.19, "PRO": 115.13, "SER": 105.09,
    "THR": 119.12, "TRP": 204.23, "TYR": 181.19, "VAL": 117.15,
}

HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
CHARGED = {"ASP", "GLU", "LYS", "ARG", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
POSITIVE_AA = {"LYS", "ARG", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

MCCE_TO_STD = {"CYD": "CYS", "NTG": None, "CTR": None, "NTR": None}

COLUMNS = [
    "pdb",
    # A. Structural
    "n_residues", "n_chains", "seq_length", "molecular_weight",
    "radius_of_gyration", "asphericity", "contact_order",
    "n_helix_residues", "n_sheet_residues", "frac_helix", "frac_sheet", "frac_coil",
    "n_disulfide", "total_sas", "mean_frac_acc", "frac_buried",
    # B. Chemistry / Sequence
    "gravy", "frac_hydrophobic", "frac_charged", "frac_polar",
    "frac_aromatic", "frac_positive", "frac_negative",
    "charge_asymmetry",
    # C. MCCE4 Electrostatics
    "net_charge_pH7", "pI", "charge_pH4", "charge_pH10",
    "buffering_capacity", "charge_regulation",
    "n_ionizable",
    "mean_pka_shift", "std_pka_shift", "max_abs_pka_shift",
    "n_large_shift_gt2", "mean_acid_shift", "mean_base_shift",
    "mean_hill_coeff", "max_hill_coeff",
    "dG_folded_elec",
    "backbone_dipole_D", "ionizable_dipole_D", "full_dipole_D",
    # D. Pairwise energies
    "respair_sum_vdw", "respair_sum_ele", "respair_sum_pairwise",
    "respair_mean_ele", "respair_n_significant",
    "n_salt_bridges",
    # E. Conformer energies
    "sum_epol", "sum_dsolv", "mean_dsolv", "mean_epol",
    "dsolv_per_buried_ion",
    # F. pKa energy decomposition
    "pke_mean_ebkb", "pke_mean_dsol", "pke_mean_residues",
    "pke_mean_TS", "pke_sum_total",
    # G. Entropy
    "conf_entropy_pH7",
    # H. Conformer diversity
    "n_active_conformers_pH7",
]


# ── A. Structural features ──────────────────────────────────────────────────

def _normalize_restype(restype):
    if restype in MCCE_TO_STD:
        return MCCE_TO_STD[restype]
    return restype if restype in STANDARD_RES else None


def count_chains_and_residues(acc_residues):
    chains = set()
    n_res = 0
    seq_len = 0
    for key, (restype, resid, _, _) in acc_residues.items():
        n_res += 1
        if resid:
            chains.add(resid[0])
        std = _normalize_restype(restype)
        if std is not None:
            seq_len += 1
    return len(chains), n_res, seq_len


def estimate_molecular_weight(acc_residues):
    total = 0.0
    count = 0
    for key, (restype, resid, _, _) in acc_residues.items():
        std = _normalize_restype(restype)
        if std and std in RESIDUE_MW:
            total += RESIDUE_MW[std]
            count += 1
    if count < 2:
        return total
    return total - (count - 1) * 18.015


def compute_radius_of_gyration(step2_path):
    atoms = _parse_backbone_ca(step2_path)
    if len(atoms) < 2:
        return None
    n = len(atoms)
    cx = sum(a[2] for a in atoms) / n
    cy = sum(a[3] for a in atoms) / n
    cz = sum(a[4] for a in atoms) / n
    sq_sum = sum((a[2]-cx)**2 + (a[3]-cy)**2 + (a[4]-cz)**2 for a in atoms)
    return math.sqrt(sq_sum / n)


def _parse_backbone_ca(step2_path):
    """Parse backbone CA coordinates and sequence numbers from step2_out.pdb."""
    atoms = []
    with open(step2_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if "BK" not in line[80:]:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                chain = line[21]
                seqnum = int(line[22:26])
                atoms.append((chain, seqnum, x, y, z))
            except (ValueError, IndexError):
                continue
    return atoms


def compute_asphericity(step2_path):
    """Asphericity from the gyration tensor eigenvalues.
    0 = perfect sphere, 1 = maximally elongated (rod).
    """
    atoms = _parse_backbone_ca(step2_path)
    if len(atoms) < 3:
        return None
    n = len(atoms)
    cx = sum(a[2] for a in atoms) / n
    cy = sum(a[3] for a in atoms) / n
    cz = sum(a[4] for a in atoms) / n
    sxx = syy = szz = sxy = sxz = syz = 0.0
    for _, _, x, y, z in atoms:
        dx, dy, dz = x - cx, y - cy, z - cz
        sxx += dx * dx
        syy += dy * dy
        szz += dz * dz
        sxy += dx * dy
        sxz += dx * dz
        syz += dy * dz
    sxx /= n; syy /= n; szz /= n; sxy /= n; sxz /= n; syz /= n
    trace = sxx + syy + szz
    if trace < 1e-8:
        return 0.0
    # eigenvalues via numpy (only 3x3 matrix)
    try:
        import numpy as np
        S = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
        eigvals = sorted(np.linalg.eigvalsh(S))
        l1, l2, l3 = eigvals
        return 1.0 - 3.0 * (l1*l2 + l1*l3 + l2*l3) / (l1 + l2 + l3)**2
    except Exception:
        return None


def compute_contact_order(step2_path, contact_cutoff=8.0):
    """Relative contact order: mean sequence separation of CA-CA contacts
    (within cutoff) divided by sequence length. Higher = more long-range contacts.
    """
    atoms = _parse_backbone_ca(step2_path)
    if len(atoms) < 5:
        return None
    total_sep = 0
    n_contacts = 0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if atoms[i][0] != atoms[j][0]:
                continue
            seq_sep = abs(atoms[i][1] - atoms[j][1])
            if seq_sep < 3:
                continue
            dx = atoms[i][2] - atoms[j][2]
            dy = atoms[i][3] - atoms[j][3]
            dz = atoms[i][4] - atoms[j][4]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= contact_cutoff:
                total_sep += seq_sep
                n_contacts += 1
    if n_contacts == 0:
        return None
    return (total_sep / n_contacts) / len(atoms)


def parse_secondary_structure(prot_pdb_path):
    n_helix = 0
    n_sheet = 0
    n_ssbond = 0
    with open(prot_pdb_path) as f:
        for line in f:
            if line.startswith("HELIX"):
                try:
                    init_seq = int(line[21:25])
                    end_seq = int(line[33:37])
                    n_helix += max(0, end_seq - init_seq + 1)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SHEET"):
                try:
                    init_seq = int(line[22:26])
                    end_seq = int(line[33:37])
                    n_sheet += max(0, end_seq - init_seq + 1)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SSBOND"):
                n_ssbond += 1
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                break
    return {"n_helix_residues": n_helix, "n_sheet_residues": n_sheet,
            "n_disulfide": n_ssbond}


def compute_acc_features(acc_residues, burial_threshold):
    frac_vals = []
    for key, (restype, resid, abs_sas, frac_acc) in acc_residues.items():
        if frac_acc is not None:
            frac_vals.append(frac_acc)
    if not frac_vals:
        return None, None
    mean_fa = sum(frac_vals) / len(frac_vals)
    frac_buried = sum(1 for v in frac_vals if v <= burial_threshold) / len(frac_vals)
    return mean_fa, frac_buried


# ── B. Chemistry / Sequence features ────────────────────────────────────────

def compute_gravy(acc_residues):
    total = 0.0
    count = 0
    for key, (restype, resid, _, _) in acc_residues.items():
        std = _normalize_restype(restype)
        if std and std in KYTE_DOOLITTLE:
            total += KYTE_DOOLITTLE[std]
            count += 1
    return total / count if count > 0 else None


def compute_composition_fractions(acc_residues):
    types = []
    for key, (restype, resid, _, _) in acc_residues.items():
        std = _normalize_restype(restype)
        if std and std in STANDARD_RES:
            types.append(std)
    n = len(types)
    if n == 0:
        return {}
    n_pos = sum(1 for t in types if t in POSITIVE_AA)
    n_neg = sum(1 for t in types if t in NEGATIVE_AA)
    return {
        "frac_hydrophobic": sum(1 for t in types if t in HYDROPHOBIC) / n,
        "frac_charged": sum(1 for t in types if t in CHARGED) / n,
        "frac_polar": sum(1 for t in types if t in POLAR) / n,
        "frac_aromatic": sum(1 for t in types if t in AROMATIC) / n,
        "frac_positive": n_pos / n,
        "frac_negative": n_neg / n,
        "charge_asymmetry": (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0,
    }


# ── C. MCCE4 Electrostatic features ─────────────────────────────────────────

def parse_head3_pka0(filepath):
    """Extract pKa0 per residue key from head3.lst, matching analyze_folding_energy.py logic."""
    pka0_map = {}
    with open(filepath) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                conformer = parts[1]
                pka0_str = parts[6]
                if pka0_str in ("0.000", "0.00", "0"):
                    continue
                pka0 = float(pka0_str)
                if len(conformer) >= 6:
                    modified = conformer[:4] + conformer[5:]
                    residue_key = modified.split("_")[0] + "_"
                    pka0_map[residue_key] = pka0
            except (ValueError, IndexError):
                continue
    return pka0_map


def parse_pk_out_full(filepath):
    """Parse pK.out returning pKa, Hill coefficient, and chi2 per residue."""
    residues = []
    with open(filepath, errors="replace") as f:
        f.readline()
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            label = line[:14].strip()
            rest = line[14:].split()
            if not rest:
                continue
            sign = "+" if "+" in label else "-"
            restype = label.split(sign)[0].split("-")[0]
            resid = label.replace(restype + sign, "").rstrip("_")
            pka_str = rest[0]
            if pka_str.startswith(">") or pka_str.startswith("<"):
                pka_val = None
            else:
                try:
                    pka_val = float(pka_str)
                except ValueError:
                    pka_val = None
            try:
                hill = float(rest[1]) if len(rest) > 1 else None
            except ValueError:
                hill = None
            residues.append((restype, resid, pka_val, pka_str, hill))
    return residues


def compute_pka_and_dG(pk_path, head3_path):
    """Compute pKa shift stats, Hill coefficients, and dG_folded."""
    pk_residues = parse_pk_out_full(pk_path)
    head3_pka0 = parse_head3_pka0(head3_path)

    all_shifts = []
    acid_shifts = []
    base_shifts = []
    hill_coeffs = []
    dG_folded = 0.0
    n_dG_residues = 0
    n_ionizable = 0

    for restype, resid, pka_val, pka_str, hill in pk_residues:
        if restype not in IONIZABLE_RES:
            continue
        n_ionizable += 1
        if hill is not None:
            hill_coeffs.append(hill)
        if pka_val is None:
            continue

        pka0 = PKA0.get(restype)
        if pka0 is not None:
            shift = pka_val - pka0
            all_shifts.append(shift)
            if restype in ACIDS:
                acid_shifts.append(shift)
            elif restype in BASES:
                base_shifts.append(shift)

        sign = "-" if restype in ACIDS else "+"
        raw_key = f"{restype}{sign}{resid}_"
        h3_pka0 = head3_pka0.get(raw_key)
        if h3_pka0 is not None:
            coeff = RT_LN10 if restype in ACIDS else -RT_LN10
            dG_folded += coeff * (pka_val - h3_pka0)
            n_dG_residues += 1

    result = {"n_ionizable": n_ionizable, "dG_folded_elec": dG_folded if n_dG_residues > 0 else None}

    if all_shifts:
        mean_s = sum(all_shifts) / len(all_shifts)
        result["mean_pka_shift"] = mean_s
        result["std_pka_shift"] = (sum((s - mean_s)**2 for s in all_shifts) / len(all_shifts)) ** 0.5
        result["max_abs_pka_shift"] = max(abs(s) for s in all_shifts)
        result["n_large_shift_gt2"] = sum(1 for s in all_shifts if abs(s) > 2.0)
    if acid_shifts:
        result["mean_acid_shift"] = sum(acid_shifts) / len(acid_shifts)
    if base_shifts:
        result["mean_base_shift"] = sum(base_shifts) / len(base_shifts)
    if hill_coeffs:
        result["mean_hill_coeff"] = sum(hill_coeffs) / len(hill_coeffs)
        result["max_hill_coeff"] = max(hill_coeffs)

    return result


# ── D. Pairwise energies (respair.lst) ──────────────────────────────────────

def parse_respair(filepath):
    sum_vdw = 0.0
    sum_ele = 0.0
    sum_pw = 0.0
    nonzero_ele = []
    n_significant = 0
    n_salt_bridges = 0
    salt_bridge_pairs = set()
    with open(filepath) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                vdw = float(parts[2])
                ele = float(parts[3])
                pw = float(parts[4])
                crg = float(parts[5])
            except (ValueError, IndexError):
                continue
            sum_vdw += vdw
            sum_ele += ele
            sum_pw += pw
            if abs(ele) > 0.001:
                nonzero_ele.append(ele)
            if abs(pw) > 0.5:
                n_significant += 1
            res1, res2 = parts[0], parts[1]
            is_charged_pair = (
                ("+" in res1 or "-" in res1) and
                abs(crg) > 0.9 and
                ele < -1.0
            )
            if is_charged_pair:
                pair_key = tuple(sorted((res1, res2)))
                salt_bridge_pairs.add(pair_key)
    return {
        "respair_sum_vdw": sum_vdw,
        "respair_sum_ele": sum_ele,
        "respair_sum_pairwise": sum_pw,
        "respair_mean_ele": sum(nonzero_ele) / len(nonzero_ele) if nonzero_ele else 0.0,
        "respair_n_significant": n_significant,
        "n_salt_bridges": len(salt_bridge_pairs),
    }


# ── E. Conformer energies (head3.lst) ───────────────────────────────────────

def parse_head3_energies(filepath):
    """Parse head3.lst by whitespace splitting.
    Header: iConf CONFORMER FL occ crg Em0 pKa0 ne nH vdw0 vdw1 tors epol dsolv extra history
    Index:    0      1      2   3   4   5   6   7  8   9   10   11   12   13    14    15
    """
    sum_epol = 0.0
    sum_dsolv = 0.0
    count = 0
    with open(filepath) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                epol = float(parts[12])
                dsolv = float(parts[13])
            except (ValueError, IndexError):
                continue
            sum_epol += epol
            sum_dsolv += dsolv
            count += 1
    if count == 0:
        return {}
    return {
        "sum_epol": sum_epol,
        "sum_dsolv": sum_dsolv,
        "mean_dsolv": sum_dsolv / count,
        "mean_epol": sum_epol / count,
    }


# ── F. pKa energy decomposition (pK_extended.out) ───────────────────────────

def parse_pk_extended(filepath):
    ebkb_vals = []
    dsol_vals = []
    res_vals = []
    ts_vals = []
    total_vals = []
    with open(filepath, errors="replace") as f:
        f.readline()
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            rest = line[14:].split()
            if len(rest) < 14:
                continue
            try:
                pka_str = rest[0].lstrip("<>")
                float(pka_str)
            except ValueError:
                continue
            try:
                ebkb_vals.append(float(rest[6]))
                dsol_vals.append(float(rest[7]))
                res_vals.append(float(rest[12]))
                ts_vals.append(float(rest[11]))
                total_vals.append(float(rest[13]))
            except (ValueError, IndexError):
                continue
    if not ebkb_vals:
        return {}
    n = len(ebkb_vals)
    return {
        "pke_mean_ebkb": sum(ebkb_vals) / n,
        "pke_mean_dsol": sum(dsol_vals) / n,
        "pke_mean_residues": sum(res_vals) / n,
        "pke_mean_TS": sum(ts_vals) / n,
        "pke_sum_total": sum(total_vals),
    }


# ── G. Entropy (entropy.out) ────────────────────────────────────────────────

def parse_entropy_at_ph(filepath, target_ph):
    with open(filepath) as f:
        header = f.readline().split()
        if header and header[0].lower() == "ph":
            header = header[1:]
        ph_vals = [float(p) for p in header]
        try:
            col_idx = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - target_ph))
        except ValueError:
            return None

        total_entropy = 0.0
        for line in f:
            parts = line.split()
            if len(parts) < col_idx + 2:
                continue
            try:
                total_entropy += float(parts[col_idx + 1])
            except (ValueError, IndexError):
                continue
    return total_entropy


# ── H. Conformer diversity (fort.38) ────────────────────────────────────────

def count_active_conformers(filepath, target_ph, occ_threshold=0.01):
    """Count conformers with occupancy > threshold at target pH."""
    with open(filepath) as f:
        header = f.readline().split()
        if header and header[0].lower() == "ph":
            header = header[1:]
        ph_vals = [float(p) for p in header]
        col_idx = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - target_ph))

        n_active = 0
        for line in f:
            parts = line.split()
            if len(parts) < col_idx + 2:
                continue
            try:
                occ = float(parts[col_idx + 1])
                if occ > occ_threshold:
                    n_active += 1
            except (ValueError, IndexError):
                continue
    return n_active


# ── Master extraction ────────────────────────────────────────────────────────

def extract_features(pdb_dir, pdb_name, target_ph, burial_threshold, dipole_funcs):
    row = {"pdb": pdb_name}

    acc_path = os.path.join(pdb_dir, "acc.res")
    crg_path = os.path.join(pdb_dir, "sum_crg.out")
    pk_path = os.path.join(pdb_dir, "pK.out")
    head3_path = os.path.join(pdb_dir, "head3.lst")
    prot_path = os.path.join(pdb_dir, "prot.pdb")
    step2_path = os.path.join(pdb_dir, "step2_out.pdb")
    respair_path = os.path.join(pdb_dir, "respair.lst")
    pke_path = os.path.join(pdb_dir, "pK_extended.out")
    entropy_path = os.path.join(pdb_dir, "entropy.out")

    # A. Structural
    if os.path.isfile(acc_path):
        acc_residues, total_sas = parse_acc_res(acc_path)
        n_chains, n_res, seq_len = count_chains_and_residues(acc_residues)
        mw = estimate_molecular_weight(acc_residues)
        mean_fa, frac_bur = compute_acc_features(acc_residues, burial_threshold)
        row.update({
            "n_residues": n_res, "n_chains": n_chains, "seq_length": seq_len,
            "molecular_weight": f"{mw:.1f}",
            "total_sas": f"{total_sas:.2f}",
            "mean_frac_acc": f"{mean_fa:.4f}" if mean_fa is not None else "",
            "frac_buried": f"{frac_bur:.4f}" if frac_bur is not None else "",
        })

        # B. Chemistry / Sequence
        gravy = compute_gravy(acc_residues)
        row["gravy"] = f"{gravy:.4f}" if gravy is not None else ""
        row.update({k: f"{v:.4f}" for k, v in compute_composition_fractions(acc_residues).items()})
    else:
        acc_residues = None

    if os.path.isfile(step2_path):
        rg = compute_radius_of_gyration(step2_path)
        row["radius_of_gyration"] = f"{rg:.3f}" if rg is not None else ""
        asp = compute_asphericity(step2_path)
        row["asphericity"] = f"{asp:.4f}" if asp is not None else ""
        co = compute_contact_order(step2_path)
        row["contact_order"] = f"{co:.4f}" if co is not None else ""

    if os.path.isfile(prot_path):
        ss = parse_secondary_structure(prot_path)
        row.update(ss)
        seq_len = row.get("seq_length", 0)
        if seq_len and seq_len > 0:
            row["frac_helix"] = f"{ss['n_helix_residues'] / seq_len:.4f}"
            row["frac_sheet"] = f"{ss['n_sheet_residues'] / seq_len:.4f}"
            frac_coil = max(0.0, 1.0 - ss["n_helix_residues"] / seq_len - ss["n_sheet_residues"] / seq_len)
            row["frac_coil"] = f"{frac_coil:.4f}"

    # C. Electrostatics
    if os.path.isfile(crg_path):
        ph_values, net_charges = parse_sum_crg(crg_path)
        q7 = get_charge_at_ph(ph_values, net_charges, target_ph)
        pi = calc_pi(ph_values, net_charges)
        q4 = get_charge_at_ph(ph_values, net_charges, 4.0)
        q10 = get_charge_at_ph(ph_values, net_charges, 10.0)
        q6 = get_charge_at_ph(ph_values, net_charges, 6.0)
        q8 = get_charge_at_ph(ph_values, net_charges, 8.0)
        buf_cap = abs(q6 - q8) / 2.0 if q6 is not None and q8 is not None else None
        row["net_charge_pH7"] = f"{q7:.2f}" if q7 is not None else ""
        row["pI"] = f"{pi:.2f}" if pi is not None else ""
        row["charge_pH4"] = f"{q4:.2f}" if q4 is not None else ""
        row["charge_pH10"] = f"{q10:.2f}" if q10 is not None else ""
        row["buffering_capacity"] = f"{buf_cap:.3f}" if buf_cap is not None else ""
        q_lo = get_charge_at_ph(ph_values, net_charges, target_ph - 0.5)
        q_hi = get_charge_at_ph(ph_values, net_charges, target_ph + 0.5)
        if q_lo is not None and q_hi is not None:
            row["charge_regulation"] = f"{abs(q_lo - q_hi):.3f}"

    if os.path.isfile(pk_path) and os.path.isfile(head3_path):
        pka_feats = compute_pka_and_dG(pk_path, head3_path)
        for k in ("n_ionizable", "n_large_shift_gt2"):
            if k in pka_feats:
                row[k] = pka_feats[k]
        for k in ("mean_pka_shift", "std_pka_shift", "max_abs_pka_shift",
                   "mean_acid_shift", "mean_base_shift",
                   "mean_hill_coeff", "max_hill_coeff",
                   "dG_folded_elec"):
            if k in pka_feats:
                row[k] = f"{pka_feats[k]:.3f}"

    if dipole_funcs:
        bb, ion, full = _compute_dipole(pdb_dir, target_ph, dipole_funcs)
        row["backbone_dipole_D"] = f"{bb:.2f}" if bb is not None else ""
        row["ionizable_dipole_D"] = f"{ion:.2f}" if ion is not None else ""
        row["full_dipole_D"] = f"{full:.2f}" if full is not None else ""

    # D. Pairwise energies
    if os.path.isfile(respair_path):
        rp = parse_respair(respair_path)
        for k in ("respair_sum_vdw", "respair_sum_ele", "respair_sum_pairwise", "respair_mean_ele"):
            row[k] = f"{rp[k]:.3f}"
        row["respair_n_significant"] = rp["respair_n_significant"]
        row["n_salt_bridges"] = rp["n_salt_bridges"]

    # E. Conformer energies
    if os.path.isfile(head3_path):
        h3e = parse_head3_energies(head3_path)
        for k in ("sum_epol", "sum_dsolv", "mean_dsolv", "mean_epol"):
            if k in h3e:
                row[k] = f"{h3e[k]:.3f}"
        if "sum_dsolv" in h3e and acc_residues:
            n_buried_ion = sum(
                1 for key, (rt, rid, asas, fa) in acc_residues.items()
                if rt in IONIZABLE_RES and fa is not None and fa <= burial_threshold
            )
            if n_buried_ion > 0:
                row["dsolv_per_buried_ion"] = f"{h3e['sum_dsolv'] / n_buried_ion:.3f}"

    # F. pKa energy decomposition
    if os.path.isfile(pke_path):
        pke = parse_pk_extended(pke_path)
        for k in ("pke_mean_ebkb", "pke_mean_dsol", "pke_mean_residues", "pke_mean_TS"):
            if k in pke:
                row[k] = f"{pke[k]:.4f}"
        if "pke_sum_total" in pke:
            row["pke_sum_total"] = f"{pke['pke_sum_total']:.3f}"

    # G. Entropy
    if os.path.isfile(entropy_path):
        ent = parse_entropy_at_ph(entropy_path, target_ph)
        row["conf_entropy_pH7"] = f"{ent:.4f}" if ent is not None else ""

    # H. Conformer diversity
    fort38_path = os.path.join(pdb_dir, "fort.38")
    if os.path.isfile(fort38_path):
        n_active = count_active_conformers(fort38_path, target_ph)
        row["n_active_conformers_pH7"] = n_active

    return row


# ── Plotting ─────────────────────────────────────────────────────────────────

FIG1_NAME = "features_structural.png"
FIG2_NAME = "features_electrostatic.png"
FIG3_NAME = "features_correlation.png"


def _load_csv(path):
    import numpy as np
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _col(rows, key):
    import numpy as np
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (ValueError, KeyError):
            pass
    return np.array(vals)


def _panel_label(ax, label):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


def run_plots(csv_path, outdir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    os.makedirs(outdir, exist_ok=True)
    rows = _load_csv(csv_path)
    sns.set_theme(style="whitegrid", font_scale=0.90)
    n = len(rows)

    # ── Figure 1: Structural & Chemical Properties (3×2) ────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.subplots_adjust(wspace=0.30, hspace=0.38, left=0.06, right=0.97,
                         top=0.94, bottom=0.08)
    fig1.suptitle(f"Structural & Chemical Properties  (n = {n} proteins)",
                  fontsize=14, fontweight="bold", y=0.98)

    # A: MW vs seq_length
    ax = axes[0, 0]
    sl = _col(rows, "seq_length")
    mw = _col(rows, "molecular_weight")
    ax.scatter(sl, mw / 1000, s=25, alpha=0.7, color="steelblue", edgecolors="gray", linewidth=0.3)
    ax.set_xlabel("Sequence Length (residues)")
    ax.set_ylabel("Molecular Weight (kDa)")
    ax.set_title("Molecular Weight vs Sequence Length")
    _panel_label(ax, "A")

    # B: Rg vs MW (log-log should show power law)
    ax = axes[0, 1]
    rg = _col(rows, "radius_of_gyration")
    mw2 = _col(rows, "molecular_weight")
    ax.scatter(mw2 / 1000, rg, s=25, alpha=0.7, color="teal", edgecolors="gray", linewidth=0.3)
    ax.set_xlabel("Molecular Weight (kDa)")
    ax.set_ylabel("Radius of Gyration (Å)")
    ax.set_title("Radius of Gyration vs Molecular Weight")
    _panel_label(ax, "B")

    # C: Secondary structure stacked histogram
    ax = axes[0, 2]
    fh = _col(rows, "frac_helix")
    fs = _col(rows, "frac_sheet")
    fc = _col(rows, "frac_coil")
    x = np.arange(n)
    order = np.argsort(fh)[::-1]
    ax.bar(x, fh[order], color="salmon", label="Helix", width=1.0)
    ax.bar(x, fs[order], bottom=fh[order], color="cornflowerblue", label="Sheet", width=1.0)
    ax.bar(x, fc[order], bottom=fh[order]+fs[order], color="lightgray", label="Coil", width=1.0)
    ax.set_xlabel("Proteins (sorted by helix content)")
    ax.set_ylabel("Fraction")
    ax.set_title("Secondary Structure Composition")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.8, loc="upper right")
    ax.set_xticks([])
    _panel_label(ax, "C")

    # D: GRAVY distribution
    ax = axes[1, 0]
    gravy = _col(rows, "gravy")
    ax.hist(gravy, bins=20, color="olivedrab", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Hydrophilic | Hydrophobic")
    mean_g = np.mean(gravy)
    ax.axvline(mean_g, color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"mean = {mean_g:.2f}")
    ax.set_xlabel("GRAVY Score")
    ax.set_ylabel("Number of Proteins")
    ax.set_title("Hydrophobicity (Kyte-Doolittle GRAVY)")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "D")

    # E: Asphericity distribution
    ax = axes[1, 1]
    asp = _col(rows, "asphericity")
    ax.hist(asp, bins=20, color="mediumpurple", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(asp), color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"mean = {np.mean(asp):.2f}")
    ax.set_xlabel("Asphericity (0 = sphere, 1 = rod)")
    ax.set_ylabel("Number of Proteins")
    ax.set_title("Shape Asphericity")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "E")

    # F: Contact order distribution
    ax = axes[1, 2]
    co = _col(rows, "contact_order")
    ax.hist(co, bins=20, color="darkcyan", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(co), color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"mean = {np.mean(co):.3f}")
    ax.set_xlabel("Relative Contact Order")
    ax.set_ylabel("Number of Proteins")
    ax.set_title("Contact Order (long-range vs local contacts)")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "F")

    out1 = os.path.join(outdir, FIG1_NAME)
    fig1.savefig(out1, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig1)

    # ── Figure 2: Electrostatic Properties (3×2) ───────────────────────
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.subplots_adjust(wspace=0.30, hspace=0.38, left=0.06, right=0.97,
                         top=0.94, bottom=0.08)
    fig2.suptitle(f"Electrostatic Properties from MCCE4  (n = {n} proteins)",
                  fontsize=14, fontweight="bold", y=0.98)

    # A: dG_folded_elec histogram
    ax = axes[0, 0]
    dg = _col(rows, "dG_folded_elec")
    ax.hist(dg, bins=20, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(np.mean(dg), color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"mean = {np.mean(dg):.1f}")
    ylim = ax.get_ylim()
    ax.axvspan(ax.get_xlim()[0], 0, color="green", alpha=0.05)
    ax.axvspan(0, ax.get_xlim()[1], color="red", alpha=0.05)
    ax.set_ylim(ylim)
    ax.set_xlabel("ΔG_folded (kcal/mol)")
    ax.set_ylabel("Number of Proteins")
    ax.set_title("Electrostatic Folding Energy")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "A")

    # B: Net charge vs pI
    ax = axes[0, 1]
    qvals = _col(rows, "net_charge_pH7")
    pivals = _col(rows, "pI")
    colors_sc = ["salmon" if q > 0 else "cornflowerblue" for q in qvals]
    ax.scatter(pivals, qvals, c=colors_sc, s=25, alpha=0.7, edgecolors="gray", linewidth=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(7.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6, label="pH 7")
    ax.set_xlabel("Isoelectric Point (pI)")
    ax.set_ylabel("Net Charge at pH 7")
    ax.set_title("Net Charge vs Isoelectric Point")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "B")

    # C: Buffering capacity vs charge regulation
    ax = axes[0, 2]
    buf = _col(rows, "buffering_capacity")
    creg = _col(rows, "charge_regulation")
    ax.scatter(buf, creg, s=25, alpha=0.7, color="darkorange", edgecolors="gray", linewidth=0.3)
    ax.set_xlabel("Buffering Capacity  |ΔQ| / ΔpH")
    ax.set_ylabel("Charge Regulation  |Q(pH±0.5)|")
    ax.set_title("Charge Buffering vs Regulation")
    _panel_label(ax, "C")

    # D: Salt bridges vs sequence length
    ax = axes[1, 0]
    sb = _col(rows, "n_salt_bridges")
    sl2 = _col(rows, "seq_length")
    ax.scatter(sl2, sb, s=25, alpha=0.7, color="firebrick", edgecolors="gray", linewidth=0.3)
    ax.set_xlabel("Sequence Length (residues)")
    ax.set_ylabel("Number of Salt Bridges")
    ax.set_title("Salt Bridges vs Protein Size")
    _panel_label(ax, "D")

    # E: Dipole vs sequence length
    ax = axes[1, 1]
    dip = _col(rows, "full_dipole_D")
    sl3 = _col(rows, "seq_length")
    if len(dip) > 0:
        ax.scatter(sl3[:len(dip)], dip, s=25, alpha=0.7, color="mediumpurple",
                   edgecolors="gray", linewidth=0.3)
    ax.set_xlabel("Sequence Length (residues)")
    ax.set_ylabel("Dipole Moment (Debye)")
    ax.set_title("Protein Dipole vs Size")
    _panel_label(ax, "E")

    # F: Desolvation per buried ionizable
    ax = axes[1, 2]
    dsolv_bi = _col(rows, "dsolv_per_buried_ion")
    ax.hist(dsolv_bi, bins=20, color="indianred", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(dsolv_bi), color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"mean = {np.mean(dsolv_bi):.1f}")
    ax.set_xlabel("Desolvation per Buried Ionizable (kcal/mol)")
    ax.set_ylabel("Number of Proteins")
    ax.set_title("Desolvation Penalty per Buried Charge")
    ax.legend(fontsize=8, framealpha=0.8)
    _panel_label(ax, "F")

    out2 = os.path.join(outdir, FIG2_NAME)
    fig2.savefig(out2, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # ── Figure 3: Feature Correlation Heatmap ──────────────────────────
    corr_features = [
        "seq_length", "molecular_weight", "radius_of_gyration", "asphericity",
        "contact_order", "frac_helix", "frac_sheet", "n_disulfide",
        "total_sas", "frac_buried",
        "gravy", "frac_hydrophobic", "frac_charged", "charge_asymmetry",
        "net_charge_pH7", "pI", "buffering_capacity", "charge_regulation",
        "mean_pka_shift", "dG_folded_elec",
        "full_dipole_D", "n_salt_bridges",
        "respair_sum_ele", "sum_dsolv", "dsolv_per_buried_ion",
        "pke_mean_dsol", "pke_mean_residues",
        "conf_entropy_pH7", "n_active_conformers_pH7",
    ]
    short_labels = [
        "SeqLen", "MW", "Rg", "Aspher", "ContOrd",
        "fHelix", "fSheet", "nSS",
        "SAS", "fBuried",
        "GRAVY", "fHydro", "fCharged", "ChgAsym",
        "Q(pH7)", "pI", "BufCap", "ChgReg",
        "ΔpKa", "ΔG_fold",
        "Dipole", "nSaltBr",
        "Σele", "Σdsolv", "dsolv/bur",
        "pke_dsol", "pke_res",
        "Entropy", "nConf",
    ]

    data = []
    valid_idx = []
    for j, feat in enumerate(corr_features):
        vals = _col(rows, feat)
        if len(vals) == n:
            data.append(vals)
            valid_idx.append(j)
    if len(data) < 3:
        print("Not enough numeric features for correlation heatmap", file=sys.stderr)
        return

    data = np.array(data)
    labels = [short_labels[j] for j in valid_idx]
    corr = np.corrcoef(data)

    fig3, ax = plt.subplots(figsize=(14, 12))
    fig3.subplots_adjust(left=0.14, bottom=0.14, right=0.98, top=0.94)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, linecolor="white",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                annot=True, fmt=".1f", annot_kws={"size": 6.5}, ax=ax)
    ax.set_title(f"Feature Correlation Matrix  (n = {n} proteins)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    out3 = os.path.join(outdir, FIG3_NAME)
    fig3.savefig(out3, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out3}")
    plt.close(fig3)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract comprehensive protein-level features from MCCE4 directories.")
    parser.add_argument("-d", "--directory", default=".",
                        help="Top-level directory containing PDB subdirectories; default: %(default)s")
    parser.add_argument("-o", "--outdir", default="extract_protein_features",
                        help="Output directory for CSV and figures; default: %(default)s")
    parser.add_argument("--ph", type=float, default=7.0,
                        help="Target pH for pH-dependent features; default: %(default)s")
    parser.add_argument("-b", "--burial-threshold", type=float, default=0.20,
                        help="Fractional accessibility threshold for buried; default: %(default)s")
    parser.add_argument("--plot", action="store_true",
                        help="Generate summary figures after feature extraction")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure DPI; default: %(default)s")
    args = parser.parse_args()

    topdir = args.directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "protein_features_full.csv")

    subdirs = list_pdb_dirs(topdir)
    if not subdirs:
        print(f"No PDB directories found in {topdir}", file=sys.stderr)
        sys.exit(1)

    dipole_funcs = _init_dipole()
    if dipole_funcs:
        print("Dipole computation enabled (ms_dipole found)")
    else:
        print("Warning: ms_dipole not available, skipping dipole features", file=sys.stderr)

    rows = []
    skipped = []
    n_total = len(subdirs)
    bar_width = 40
    t_start = time.time()

    for i, pdb in enumerate(subdirs, 1):
        pdb_dir = os.path.join(topdir, pdb)
        try:
            row = extract_features(pdb_dir, pdb, args.ph, args.burial_threshold, dipole_funcs)
            rows.append(row)
            status = pdb
        except Exception as e:
            skipped.append((pdb, str(e)))
            status = f"{pdb} (SKIPPED)"

        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n_total - i) / rate if rate > 0 else 0
        filled = int(bar_width * i / n_total)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = 100 * i / n_total
        sys.stderr.write(
            f"\r  {bar} {pct:5.1f}%  [{i}/{n_total}]  "
            f"{status:<6s}  {elapsed:.1f}s elapsed  ETA {eta:.0f}s  "
        )
        sys.stderr.flush()

    sys.stderr.write("\n")

    if not rows:
        print("No proteins processed successfully.", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    elapsed_total = time.time() - t_start
    print(f"Wrote {len(rows)} proteins ({len(COLUMNS)-1} features) to {csv_path}  "
          f"[{elapsed_total:.1f}s total, {elapsed_total/len(rows):.2f}s/protein]")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(p for p, _ in skipped)}")

    if args.plot:
        print("\nGenerating summary figures...")
        run_plots(csv_path, outdir, args.dpi)


if __name__ == "__main__":
    main()
