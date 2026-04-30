#!/usr/bin/env python3
"""
mcce2ml_1-features.py
=====================
Combined MCCE → ML feature pipeline.
Parses pK.out/acc.res, extracts structural features from step1_out.pdb,
computes protein-level sequence features, and outputs: ml_dataset.pkl

Usage:
    python mcce2ml_1-features.py                  # default output
    python mcce2ml_1-features.py -o my_data.pkl   # custom name

Pipeline:
    1. Parse pK.out / acc.res  →  residue-level pKa shifts & SASA
    2. Generate structural features from step1_out.pdb (shells, EPI, charges …)
    3. Extract protein-level sequence features (AAC, physicochemical, transitions)
    4. Flatten & merge into one ML-ready DataFrame  →  ml_dataset.pkl
"""

import os
import re
import argparse
import time
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch
import warnings

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PKA0_REF = {
    "NTR": 8.00, "CTR": 3.75, "ASP": 4.75, "GLU": 4.75,
    "TYR": 10.20, "ARG": 12.5, "HIS": 6.98, "LYS": 10.4,
}

TITRATABLE = set(PKA0_REF.keys())

CHARGE_MAP = {"ARG": 1, "LYS": 1, "HIS": 1, "ASP": -1, "GLU": -1}
POLAR_ELEMENTS = {"O", "N", "S"}

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}

SEQ_PKA = {
    "A": 0.0, "R": 12.5, "N": 0.0, "D": 3.65, "C": 8.18,
    "Q": 0.0, "E": 4.25, "G": 0.0, "H": 6.0, "I": 0.0,
    "L": 0.0, "K": 10.53, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 10.07, "V": 0.0,
}

FLEXIBILITY = {
    "A": 0.984, "R": 1.008, "N": 1.048, "D": 1.068, "C": 0.906,
    "Q": 1.037, "E": 1.094, "G": 1.031, "H": 0.950, "I": 0.927,
    "L": 0.935, "K": 1.102, "M": 0.952, "F": 0.915, "P": 1.049,
    "S": 1.046, "T": 0.997, "W": 0.904, "Y": 0.929, "V": 0.931,
}

CHARGE_MAP_SEQ = {"R": 1.0, "H": 0.5, "K": 1.0, "D": -1.0, "E": -1.0}
CHARGE_MAP_SEQ.update({aa: 0.0 for aa in STANDARD_AA if aa not in CHARGE_MAP_SEQ})

AROMATIC = set("FWYH")
ALIPHATIC = set("AILV")
HYDROPHOBIC_SET = set("AILMFVWP")
POLAR_SET = set("STNQYDERHKC")
POSITIVE_SET = set("RHK")
NEGATIVE_SET = set("DE")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1  –  PARSE MCCE OUTPUTS  (pK.out + acc.res)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_sasa_file(pdb_dir):
    acc_file = os.path.join(pdb_dir, "acc.res")
    sasa_map = {}
    if os.path.exists(acc_file):
        with open(acc_file) as fh:
            for line in fh:
                if not line.startswith("RES"):
                    continue
                res_type = line[5:9].strip()
                chain = line[10:11].strip()
                res_num = line[11:16].strip()
                res_chain_id = f"{chain}{res_num.zfill(4)}"
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        sasa_map[(res_type, res_chain_id)] = (float(parts[-2]), float(parts[-1]))
                    except ValueError:
                        continue
    return sasa_map


def parse_pka_files(pdb_dirs):
    all_rows = []
    for pdb in pdb_dirs:
        pk_file = os.path.join(pdb, "pK.out")
        sasa_lookup = parse_sasa_file(pdb)
        if not os.path.exists(pk_file):
            continue
        with open(pk_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or any(x in line for x in ["pH", "pKa/Em", "1000*chi2"]):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                res_id_full = parts[0]
                res_type = res_id_full[:3]
                if res_type not in TITRATABLE:
                    continue
                res_chain_id = res_id_full[4:9].replace("_", "")
                sasa_data = sasa_lookup.get((res_type, res_chain_id), (np.nan, np.nan))
                try:
                    pka_float = float(parts[1].replace(">", "").replace("<", ""))
                    pka0 = PKA0_REF.get(res_type)
                    pka_shift = pka_float - pka0 if pka0 is not None else np.nan
                except ValueError:
                    pka_float, pka_shift = np.nan, np.nan
                    pka0 = PKA0_REF.get(res_type)
                all_rows.append({
                    "PDB": pdb, "Residue": res_type, "ResChain+Id": res_chain_id,
                    "pKa0": pka0, "Target_pKa": pka_float,
                    "Target_pKa_shift": pka_shift,
                    "SASA_abs": sasa_data[0], "SASA_rel": sasa_data[1],
                })
    df = pd.DataFrame(all_rows).dropna(subset=["Target_pKa_shift"])
    print(f"[Step 1] Parsed {len(df)} titratable residues from {df['PDB'].nunique()} PDBs")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2  –  STRUCTURAL FEATURES  (from step1_out.pdb)
# ═══════════════════════════════════════════════════════════════════════════════

def get_structure_data(structure):
    all_heavy, res_ca = [], {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue):
                    continue
                rname = residue.get_resname()
                rnum = str(residue.get_id()[1]).zfill(4)
                cid = chain.get_id()
                full_id = f"{rname}{cid}{rnum}"
                if "CA" in residue:
                    res_ca[residue] = {"id": full_id, "coord": residue["CA"].get_coord()}
                for atom in residue:
                    if atom.element != "H":
                        all_heavy.append(atom)
    return all_heavy, res_ca


def generate_structural_features(df_mcce, pdb_dirs):
    parser = PDB.PDBParser(QUIET=True)
    all_rows = []
    for pdb in pdb_dirs:
        pdb_path = os.path.join(pdb, "step1_out.pdb")
        if not os.path.exists(pdb_path):
            continue
        try:
            struct = parser.get_structure(pdb, pdb_path)
            heavy_atoms, res_ca_map = get_structure_data(struct)
            ns = NeighborSearch(heavy_atoms)
            pdb_mcce = df_mcce[df_mcce["PDB"] == pdb].copy()
            pdb_mcce["MatchID"] = pdb_mcce["Residue"] + pdb_mcce["ResChain+Id"]
            pdb_mcce = pdb_mcce.set_index("MatchID")

            for target_res, target_info in res_ca_map.items():
                res_id = target_info["id"]
                if res_id not in pdb_mcce.index:
                    continue
                origin_ca = target_info["coord"]
                target_atoms = [a for a in target_res.get_atoms() if a.element != "H"]

                ld1, ld2, ld3 = set(), set(), set()
                nb6_atoms = []
                for atom in target_atoms:
                    for nr in ns.search(atom.get_coord(), 3.0, level="R"):
                        if nr != target_res: ld1.add(nr)
                    for nr in ns.search(atom.get_coord(), 6.0, level="R"):
                        if nr != target_res and nr not in ld1: ld2.add(nr)
                    for nr in ns.search(atom.get_coord(), 15.0, level="R"):
                        if nr != target_res and nr not in ld1 and nr not in ld2: ld3.add(nr)
                    nb6_atoms.extend(ns.search(atom.get_coord(), 6.0, level="A"))

                ext_atoms = list({a for a in nb6_atoms if a.get_parent() != target_res})
                ext_res = {a.get_parent() for a in ext_atoms}

                elements = [a.element for a in ext_atoms]
                c_count = elements.count("C")
                on_count = sum(1 for e in elements if e in POLAR_ELEMENTS)
                total_packing = len(ext_atoms)
                bb_count = sum(1 for a in ext_atoms if a.get_name() in {"N", "CA", "C", "O"})
                net_charge = sum(CHARGE_MAP.get(r.get_resname(), 0) for r in ext_res)

                hb_atoms = []
                for atom in target_atoms:
                    hb_atoms.extend(ns.search(atom.get_coord(), 3.0, level="A"))
                hbond_potential = sum(
                    1 for a in set(hb_atoms) if a.get_parent() != target_res and a.element in {"O", "N"}
                )

                shells_data = [[], [], []]
                epi = 0.0
                for s_idx, rset in enumerate([ld1, ld2, ld3]):
                    for nr in rset:
                        if nr not in res_ca_map: continue
                        nid = res_ca_map[nr]["id"]
                        n_coord = res_ca_map[nr]["coord"]
                        ca_dist = np.linalg.norm(n_coord - origin_ca)
                        if nid in pdb_mcce.index:
                            row = pdb_mcce.loc[nid]
                            if isinstance(row, pd.DataFrame): row = row.iloc[0]
                            p0, p_shift = row["pKa0"], row["Target_pKa_shift"]
                            shells_data[s_idx].append({"vector": n_coord - origin_ca, "pKa0": p0})
                            if pd.notnull(p_shift) and ca_dist > 0:
                                epi += float(p_shift) / (ca_dist ** 2)

                target_row = pdb_mcce.loc[res_id]
                if isinstance(target_row, pd.DataFrame): target_row = target_row.iloc[0]

                all_rows.append({
                    "PDB": pdb, "Residue_ID": res_id,
                    "Target_pKa_shift": target_row["Target_pKa_shift"],
                    "pKa0": float(target_row["pKa0"]), "SASA_rel": target_row["SASA_rel"],
                    "Polarity_Ratio": round(on_count / c_count, 3) if c_count > 0 else float(on_count),
                    "Backbone_Density": bb_count, "Total_Packing": total_packing,
                    "HBond_Potential": hbond_potential, "Net_Charge_6A": int(net_charge),
                    "LD1_count": len(shells_data[0]), "LD2_count": len(shells_data[1]),
                    "LD3_count": len(shells_data[2]), "EPI": round(epi, 3),
                })
        except Exception as e:
            print(f"  [WARN] Structural feature extraction failed for {pdb}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    print(f"[Step 2] Computed structural features for {len(df)} residues")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3  –  SEQUENCE FEATURES  (protein-level)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sequence(structure):
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    code = AA3TO1.get(residue.get_resname(), "X")
                    if code != "X":
                        seq.append(code)
    return seq


def _safe_mean(seq, lookup):
    vals = [lookup.get(aa, 0.0) for aa in seq]
    return np.mean(vals) if vals else 0.0


def compute_aac(seq):
    from collections import Counter
    n = len(seq)
    if n == 0:
        return {f"AAC_{aa}": 0.0 for aa in STANDARD_AA}
    counts = Counter(seq)
    return {f"AAC_{aa}": counts.get(aa, 0) / n for aa in STANDARD_AA}


def compute_physicochemical(seq):
    n = len(seq)
    if n == 0:
        return {k: 0.0 for k in [
            "PhysChem_hydrophobicity_mean", "PhysChem_volume_mean",
            "PhysChem_polarity_mean", "PhysChem_charge_mean",
            "PhysChem_pka_mean", "PhysChem_flexibility_mean",
            "PhysChem_aromatic_ratio", "PhysChem_aliphatic_ratio",
        ]}
    return {
        "PhysChem_hydrophobicity_mean": _safe_mean(seq, HYDROPHOBICITY),
        "PhysChem_volume_mean": _safe_mean(seq, VOLUME),
        "PhysChem_polarity_mean": _safe_mean(seq, POLARITY),
        "PhysChem_charge_mean": _safe_mean(seq, CHARGE_MAP_SEQ),
        "PhysChem_pka_mean": _safe_mean(seq, SEQ_PKA),
        "PhysChem_flexibility_mean": _safe_mean(seq, FLEXIBILITY),
        "PhysChem_aromatic_ratio": sum(1 for aa in seq if aa in AROMATIC) / n,
        "PhysChem_aliphatic_ratio": sum(1 for aa in seq if aa in ALIPHATIC) / n,
    }


def compute_transitions(seq):
    n = len(seq) - 1
    if n <= 0:
        return {
            "Transition_hydrophobic_polar": 0.0, "Transition_polar_hydrophobic": 0.0,
            "Transition_positive_negative": 0.0, "Transition_negative_positive": 0.0,
        }
    hp, ph, pn, np_ = 0, 0, 0, 0
    for i in range(n):
        a, b = seq[i], seq[i + 1]
        if a in HYDROPHOBIC_SET and b in POLAR_SET: hp += 1
        if a in POLAR_SET and b in HYDROPHOBIC_SET: ph += 1
        if a in POSITIVE_SET and b in NEGATIVE_SET: pn += 1
        if a in NEGATIVE_SET and b in POSITIVE_SET: np_ += 1
    return {
        "Transition_hydrophobic_polar": hp / n, "Transition_polar_hydrophobic": ph / n,
        "Transition_positive_negative": pn / n, "Transition_negative_positive": np_ / n,
    }


def compute_sequence_features(structure):
    seq = extract_sequence(structure)
    feats = {"SeqLen": len(seq)}
    feats.update(compute_aac(seq))
    feats.update(compute_physicochemical(seq))
    feats.update(compute_transitions(seq))
    return feats


def build_sequence_feature_table(pdb_dirs):
    parser = PDB.PDBParser(QUIET=True)
    rows = []
    for pdb in pdb_dirs:
        pdb_path = os.path.join(pdb, "step1_out.pdb")
        if not os.path.exists(pdb_path): continue
        try:
            struct = parser.get_structure(pdb, pdb_path)
            feats = compute_sequence_features(struct)
            feats["PDB"] = pdb
            rows.append(feats)
        except Exception as e:
            print(f"  [WARN] Sequence features failed for {pdb}: {e}")
    df = pd.DataFrame(rows)
    print(f"[Step 3] Extracted sequence features for {len(df)} PDBs "
          f"({len(df.columns) - 1} features per protein)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4  –  FLATTEN & MERGE  →  ml_dataset.pkl
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_row(row):
    res_id_raw = str(row["Residue_ID"])
    match = re.match(r"([A-Z]{3})(.*)", res_id_raw)
    if match:
        res_type = match.group(1)
        res_id_combined = f"{res_type}{match.group(2).replace('_', '').replace('-', '')}"
    else:
        res_type = "UNK"
        res_id_combined = res_id_raw
    return res_type, res_id_combined


def build_ml_dataset(df_struct, df_seq):
    df = df_struct.merge(df_seq, on="PDB", how="left")
    res_info = df.apply(flatten_row, axis=1, result_type="expand")
    df["resi_type"] = res_info[0]
    df["resi+chainId+resid"] = res_info[1]

    id_cols = ["PDB", "resi+chainId+resid", "resi_type"]
    structural_cols = [
        "pKa0", "SASA_rel", "Polarity_Ratio", "Backbone_Density",
        "Total_Packing", "HBond_Potential", "Net_Charge_6A", "EPI",
        "LD1_count", "LD2_count", "LD3_count",
    ]
    seq_cols = [c for c in df_seq.columns if c != "PDB"]
    target_col = ["Target_pKa_shift"]
    final_cols = [c for c in id_cols + structural_cols + seq_cols + target_col if c in df.columns]

    ml_df = df[final_cols].dropna(subset=["Target_pKa_shift"]).copy()
    return ml_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mcce2ml_1-features: parse → structural → sequence → ml_dataset.pkl",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-o", "--output", default="ml_dataset.pkl",
                        help="Output pickle filename (default: ml_dataset.pkl)")
    args = parser.parse_args()
    t0 = time.time()

    pdb_dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and len(d) == 4])
    if not pdb_dirs:
        print("ERROR: No 4-letter PDB directories found in the current directory.")
        return

    print(f"Found {len(pdb_dirs)} PDB directories: {', '.join(pdb_dirs[:10])}"
          f"{'…' if len(pdb_dirs) > 10 else ''}\n")

    # Step 1 — Parse MCCE pKa outputs
    t1 = time.time()
    df_mcce = parse_pka_files(pdb_dirs)
    if df_mcce.empty:
        print("ERROR: No pKa data parsed. Aborting."); return
    print(f"         ({time.time() - t1:.1f}s)")

    # Step 2 — Structural features
    t2 = time.time()
    df_struct = generate_structural_features(df_mcce, pdb_dirs)
    if df_struct.empty:
        print("ERROR: No structural features generated. Aborting."); return
    print(f"         ({time.time() - t2:.1f}s)")

    # Step 3 — Sequence features (protein-level)
    t3 = time.time()
    df_seq = build_sequence_feature_table(pdb_dirs)
    print(f"         ({time.time() - t3:.1f}s)")

    # Step 4 — Merge & flatten
    t4 = time.time()
    ml_df = build_ml_dataset(df_struct, df_seq)
    ml_df.to_pickle(args.output)
    print(f"[Step 4] Merged & saved ({time.time() - t4:.1f}s)")

    n_struct = len([c for c in ml_df.columns if c not in
                    ["PDB", "resi+chainId+resid", "resi_type", "Target_pKa_shift"]
                    and not c.startswith("AAC_") and not c.startswith("PhysChem_")
                    and not c.startswith("Transition_") and c != "SeqLen"])
    n_seq = len([c for c in ml_df.columns if c.startswith(("AAC_", "PhysChem_", "Transition_"))
                 or c == "SeqLen"])

    # Categorize columns for display
    id_cols = [c for c in ml_df.columns if c in ["PDB", "resi+chainId+resid", "resi_type"]]
    struct_cols = [c for c in ml_df.columns if c not in id_cols + ["Target_pKa_shift"]
                   and not c.startswith("AAC_") and not c.startswith("PhysChem_")
                   and not c.startswith("Transition_") and c != "SeqLen"]
    aac_cols = sorted([c for c in ml_df.columns if c.startswith("AAC_")])
    physchem_cols = sorted([c for c in ml_df.columns if c.startswith("PhysChem_")])
    trans_cols = sorted([c for c in ml_df.columns if c.startswith("Transition_")])
    seqlen_col = ["SeqLen"] if "SeqLen" in ml_df.columns else []

    print(f"\n{'='*60}")
    print(f"SUCCESS: '{args.output}' built")
    print(f"{'='*60}")
    print(f"  Residues:            {len(ml_df)}")
    print(f"  PDBs:                {ml_df['PDB'].nunique()}")
    print(f"  Structural features: {n_struct}")
    print(f"  Sequence features:   {n_seq}")
    print(f"  Total columns:       {len(ml_df.columns)}")
    print(f"  Target:              Target_pKa_shift")
    print(f"  Mean shift:          {ml_df['Target_pKa_shift'].mean():.3f}")

    # ── RESIDUE STATISTICS ──
    print(f"\n{'-'*60}")
    print(f"RESIDUE STATISTICS")
    print(f"{'-'*60}")

    # Per-residue-type counts and shift stats
    res_counts = ml_df['resi_type'].value_counts().sort_values(ascending=False)
    total = len(ml_df)
    print(f"\n  {'Type':<6} | {'Count':>6} | {'%':>6} | {'Mean Shift':>11} | {'Std':>7} | {'Min':>7} | {'Max':>7}")
    print(f"  {'-'*62}")
    for rtype, count in res_counts.items():
        subset = ml_df[ml_df['resi_type'] == rtype]['Target_pKa_shift']
        print(f"  {rtype:<6} | {count:>6} | {count/total*100:>5.1f}% | "
              f"{subset.mean():>+11.3f} | {subset.std():>7.3f} | {subset.min():>7.2f} | {subset.max():>7.2f}")
    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<6} | {total:>6} | 100.0%")

    # Per-PDB summary
    pdb_counts = ml_df.groupby('PDB').size().sort_values(ascending=False)
    print(f"\n  PDBs: {ml_df['PDB'].nunique()}")
    print(f"  Residues per PDB:  min={pdb_counts.min()}  median={pdb_counts.median():.0f}  "
          f"max={pdb_counts.max()}  mean={pdb_counts.mean():.1f}")

    # Global pKa shift distribution
    shifts = ml_df['Target_pKa_shift']
    print(f"\n  pKa Shift Distribution:")
    print(f"    Mean:   {shifts.mean():>+.3f}")
    print(f"    Std:    {shifts.std():>.3f}")
    print(f"    Median: {shifts.median():>+.3f}")
    print(f"    Min:    {shifts.min():>+.3f}")
    print(f"    Max:    {shifts.max():>+.3f}")
    print(f"    IQR:    [{shifts.quantile(0.25):>+.3f}, {shifts.quantile(0.75):>+.3f}]")
    print(f"    |shift| ≤ 1.0:  {(shifts.abs() <= 1.0).sum():>5} ({(shifts.abs() <= 1.0).mean()*100:.1f}%)")
    print(f"    |shift| ≤ 2.0:  {(shifts.abs() <= 2.0).sum():>5} ({(shifts.abs() <= 2.0).mean()*100:.1f}%)")
    print(f"    |shift| > 2.0:  {(shifts.abs() > 2.0).sum():>5} ({(shifts.abs() > 2.0).mean()*100:.1f}%)")

    # Missing data check
    n_missing = ml_df.isnull().sum()
    cols_with_missing = n_missing[n_missing > 0]
    if not cols_with_missing.empty:
        print(f"\n  Missing Values:")
        for c, n in cols_with_missing.items():
            print(f"    {c}: {n} ({n/total*100:.1f}%)")
    else:
        print(f"\n  Missing Values: None")

    # ── FEATURE MANIFEST ──
    print(f"\n{'-'*60}")
    print(f"FEATURE MANIFEST")
    print(f"{'-'*60}")
    print(f"  Identifiers ({len(id_cols)}):")
    for c in id_cols:
        print(f"    {c}")
    print(f"\n  Structural Features ({len(struct_cols)}):")
    for c in struct_cols:
        print(f"    {c}")
    print(f"\n  Sequence — Amino Acid Composition ({len(aac_cols)}):")
    print(f"    {', '.join(aac_cols)}")
    print(f"\n  Sequence — Physicochemical ({len(physchem_cols)}):")
    for c in physchem_cols:
        print(f"    {c}")
    print(f"\n  Sequence — Transitions ({len(trans_cols)}):")
    for c in trans_cols:
        print(f"    {c}")
    if seqlen_col:
        print(f"\n  Sequence — Other ({len(seqlen_col)}):")
        print(f"    SeqLen")
    print(f"\n  Target (1):")
    print(f"    Target_pKa_shift")

    # ── TIMING ──
    elapsed = time.time() - t0
    if elapsed >= 60:
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m {secs:.1f}s"
    else:
        time_str = f"{elapsed:.1f}s"
    print(f"\n{'='*60}")
    print(f"Processing time: {time_str}")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
