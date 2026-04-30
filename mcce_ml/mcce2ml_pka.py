#!/usr/bin/env python3
"""
mcce2ml_pKa.py
==============
Predict pKa values for a single PDB structure using a trained MCCE-ML model.
Generates structural + sequence features matching mcce2ml_1-features.py,
then runs inference and optionally compares against ground truth.

Usage:
    python mcce2ml_pKa.py structure.pdb
    python mcce2ml_pKa.py structure.pdb -m MCCE_ML-models/model_rf
    python mcce2ml_pKa.py structure.pdb --all
"""

import sys
import argparse
import os
import subprocess
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (must mirror mcce2ml_1-features.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

CHARGE_MAP = {"ARG": 1, "LYS": 1, "HIS": 1, "ASP": -1, "GLU": -1}
POLAR_ELEMENTS = {'O', 'N', 'S'}
HBOND_ELEMENTS = {'O', 'N'}

PKA0_REF = {
    "ASP": 4.75, "GLU": 4.75, "ARG": 12.5, "HIS": 6.98,
    "LYS": 10.4, "TYR": 10.20, "NTR": 8.00, "CTR": 3.75,
}

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
TARGETS = {"ASP", "GLU", "ARG", "HIS", "LYS", "TYR"}

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP CLASSIFICATION  (mirrors mcce2ml_2-train.py)
# ═══════════════════════════════════════════════════════════════════════════════

_STRUCTURAL_CORE = {
    'pKa0', 'SASA_rel', 'Polarity_Ratio', 'Backbone_Density',
    'Total_Packing', 'HBond_Potential', 'Net_Charge_6A', 'EPI',
    'LD1_count', 'LD2_count', 'LD3_count',
}

# Six sub-categories for the left pie chart
GROUP_ORDER = [
    'Structural\nFeatures (11)',
    'Residue Type\n(One-Hot)',
    'Amino Acid\nComposition (20)',
    'Physicochemical\nProperties (8)',
    'Transition\nFeatures (4)',
    'Sequence\nLength (1)',
]
GROUP_PALETTE = [
    '#1b5e20',   # Structural    – dark green
    '#388e3c',   # Residue Type  – medium green
    '#4a148c',   # AAC           – deep purple
    '#01579b',   # PhysChem      – dark blue
    '#e65100',   # Transition    – deep orange
    '#888888',   # SeqLen        – grey
]

# Meta-group mapping:  sub-category -> high-level category
META_GROUPS = {
    'Structural\nFeatures (11)':       'Structure-Based',
    'Residue Type\n(One-Hot)':         'Structure-Based',
    'Amino Acid\nComposition (20)':    'Sequence-Based',
    'Physicochemical\nProperties (8)': 'Sequence-Based',
    'Transition\nFeatures (4)':        'Sequence-Based',
    'Sequence\nLength (1)':            'Sequence-Based',
}
META_PALETTE = {'Structure-Based': '#1565C0', 'Sequence-Based': '#C62828'}


def _classify_feature(feat):
    if feat in _STRUCTURAL_CORE:
        return GROUP_ORDER[0]          # Structural Features (11)
    if feat.startswith('resi_type_'):
        return GROUP_ORDER[1]          # Residue Type (One-Hot)
    if feat.startswith('AAC_'):
        return GROUP_ORDER[2]          # Amino Acid Composition (20)
    if feat.startswith('PhysChem_'):
        return GROUP_ORDER[3]          # Physicochemical Properties (8)
    if feat.startswith('Transition_'):
        return GROUP_ORDER[4]          # Transition Features (4)
    if feat == 'SeqLen':
        return GROUP_ORDER[5]          # Sequence Length (1)
    return GROUP_ORDER[5]              # fallback



# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE FEATURES  (mirrors mcce2ml_1-features.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sequence_from_structure(structure):
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    code = AA3TO1.get(residue.get_resname(), "X")
                    if code != "X": seq.append(code)
    return seq


def _safe_mean(seq, lookup):
    vals = [lookup.get(aa, 0.0) for aa in seq]
    return np.mean(vals) if vals else 0.0


def compute_sequence_features(structure):
    seq = extract_sequence_from_structure(structure)
    feats = {"SeqLen": len(seq)}
    n = len(seq)

    # AAC
    counts = Counter(seq)
    for aa in STANDARD_AA:
        feats[f"AAC_{aa}"] = counts.get(aa, 0) / n if n > 0 else 0.0

    # Physicochemical
    if n > 0:
        feats["PhysChem_hydrophobicity_mean"] = _safe_mean(seq, HYDROPHOBICITY)
        feats["PhysChem_volume_mean"] = _safe_mean(seq, VOLUME)
        feats["PhysChem_polarity_mean"] = _safe_mean(seq, POLARITY)
        feats["PhysChem_charge_mean"] = _safe_mean(seq, CHARGE_MAP_SEQ)
        feats["PhysChem_pka_mean"] = _safe_mean(seq, SEQ_PKA)
        feats["PhysChem_flexibility_mean"] = _safe_mean(seq, FLEXIBILITY)
        feats["PhysChem_aromatic_ratio"] = sum(1 for aa in seq if aa in AROMATIC) / n
        feats["PhysChem_aliphatic_ratio"] = sum(1 for aa in seq if aa in ALIPHATIC) / n
    else:
        for k in ["PhysChem_hydrophobicity_mean", "PhysChem_volume_mean",
                   "PhysChem_polarity_mean", "PhysChem_charge_mean",
                   "PhysChem_pka_mean", "PhysChem_flexibility_mean",
                   "PhysChem_aromatic_ratio", "PhysChem_aliphatic_ratio"]:
            feats[k] = 0.0

    # Transitions
    npairs = n - 1
    if npairs > 0:
        hp, ph, pn, np_ = 0, 0, 0, 0
        for i in range(npairs):
            a, b = seq[i], seq[i + 1]
            if a in HYDROPHOBIC_SET and b in POLAR_SET: hp += 1
            if a in POLAR_SET and b in HYDROPHOBIC_SET: ph += 1
            if a in POSITIVE_SET and b in NEGATIVE_SET: pn += 1
            if a in NEGATIVE_SET and b in POSITIVE_SET: np_ += 1
        feats["Transition_hydrophobic_polar"] = hp / npairs
        feats["Transition_polar_hydrophobic"] = ph / npairs
        feats["Transition_positive_negative"] = pn / npairs
        feats["Transition_negative_positive"] = np_ / npairs
    else:
        for k in ["Transition_hydrophobic_polar", "Transition_polar_hydrophobic",
                   "Transition_positive_negative", "Transition_negative_positive"]:
            feats[k] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# MCCE / STRUCTURAL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_mcce_step1(pdb_file):
    if not os.path.exists(pdb_file):
        print(f"[ERROR] PDB file {pdb_file} not found."); return False
    print(f"--> Running MCCE step1.py on {pdb_file}...")
    try:
        subprocess.run(["step1.py", pdb_file], check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] MCCE step1.py failed or not found."); return False


def parse_acc_res():
    sasa_map = {}
    if os.path.exists("acc.res"):
        with open("acc.res") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[0] == "RES":
                    sasa_map[(parts[1], f"{parts[2]}{parts[3]}")] = float(parts[5])
    return sasa_map


def generate_features_for_predict(pdb_path, sasa_lookup):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('input', pdb_path)
    except Exception:
        return pd.DataFrame()

    seq_feats = compute_sequence_features(structure)

    all_heavy, res_ca = [], {}
    backbone_names = {'N', 'CA', 'C', 'O'}

    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue): continue
                rname = residue.get_resname()
                rnum = str(residue.get_id()[1]).zfill(4)
                cid = chain.get_id()
                full_id = f"{rname}{cid}{rnum}"
                if 'CA' in residue:
                    res_ca[residue] = {'id': full_id, 'coord': residue['CA'].get_coord(),
                                       'resname': rname}
                for atom in residue:
                    if atom.element != 'H':
                        all_heavy.append(atom)

    if not res_ca: return pd.DataFrame()
    ns = NeighborSearch(all_heavy)
    rows = []

    for target_res, info in res_ca.items():
        full_id = info['id']
        res_name = info['resname']
        if res_name not in TARGETS: continue

        origin_ca = info['coord']
        chain_id, res_num = full_id[3:4], full_id[4:]
        target_atoms = [a for a in target_res.get_atoms() if a.element != 'H']

        ld1, ld2, ld3 = set(), set(), set()
        nb6_atoms = []
        for atom in target_atoms:
            for nr in ns.search(atom.get_coord(), 3.0, level='R'):
                if nr != target_res: ld1.add(nr)
            for nr in ns.search(atom.get_coord(), 6.0, level='R'):
                if nr != target_res and nr not in ld1: ld2.add(nr)
            for nr in ns.search(atom.get_coord(), 15.0, level='R'):
                if nr != target_res and nr not in ld1 and nr not in ld2: ld3.add(nr)
            nb6_atoms.extend(ns.search(atom.get_coord(), 6.0, level='A'))

        ext_atoms = list({a for a in nb6_atoms if a.get_parent() != target_res})
        ext_res = {a.get_parent() for a in ext_atoms}

        elements = [a.element for a in ext_atoms]
        c_count = elements.count('C')
        on_count = sum(1 for e in elements if e in POLAR_ELEMENTS)
        total_packing = len(ext_atoms)
        bb_count = sum(1 for a in ext_atoms if a.get_name() in backbone_names)
        net_charge = sum(CHARGE_MAP.get(r.get_resname(), 0) for r in ext_res)

        hb_atoms = []
        for atom in target_atoms:
            hb_atoms.extend(ns.search(atom.get_coord(), 3.0, level='A'))
        hbond_potential = sum(1 for a in set(hb_atoms)
                             if a.get_parent() != target_res and a.element in HBOND_ELEMENTS)

        epi = 0.0
        for rset in [ld1, ld2, ld3]:
            for nr in rset:
                if nr not in res_ca: continue
                n_coord = res_ca[nr]['coord']
                ca_dist = np.linalg.norm(n_coord - origin_ca)
                if ca_dist > 0:
                    epi += PKA0_REF.get(res_ca[nr]['resname'], 0.0) / (ca_dist ** 2)

        feat_dict = {
            'resi+chainId+resid': full_id, 'resi_type': res_name,
            'pKa0': PKA0_REF.get(res_name, 0.0),
            'SASA_rel': sasa_lookup.get((res_name, f"{chain_id}{res_num}"), 0.0),
            'Polarity_Ratio': round(on_count / c_count, 3) if c_count > 0 else 0.0,
            'Backbone_Density': bb_count, 'Total_Packing': total_packing,
            'HBond_Potential': hbond_potential, 'Net_Charge_6A': int(net_charge),
            'EPI': round(epi, 3),
            'LD1_count': len(ld1), 'LD2_count': len(ld2), 'LD3_count': len(ld3),
        }
        feat_dict.update(seq_feats)
        rows.append(feat_dict)

    return pd.DataFrame(rows)

# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# FEATURE IMPORTANCE VISUALISATION
# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═

def _add_pie_leaders(ax, wedges, labels, pcts, threshold=4.0,
                     inner_r=0.72, outer_r=1.18, text_r=1.32):
    """Label large slices inside; draw leader lines for small slices."""
    import math
    for wedge, label, pct in zip(wedges, labels, pcts):
        if pct < 0.3:
            continue
        ang    = (wedge.theta1 + wedge.theta2) / 2.0
        rad    = math.radians(ang)
        ca, sa = math.cos(rad), math.sin(rad)
        if pct >= threshold:
            ax.text(inner_r * ca, inner_r * sa, f'{pct:.1f}%',
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            ha = 'left' if ca >= 0 else 'right'
            ax.annotate(
                f'{pct:.1f}%',
                xy=(0.97 * ca, 0.97 * sa),
                xytext=((text_r + 0.08) * ca, text_r * sa),
                fontsize=8, color='#222', ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color='#666', lw=0.85,
                                connectionstyle='arc3,rad=0.0'),
            )


def _draw_meta_brackets(ax, wedges, group_names, arc_r=1.55, text_r=1.78):
    """
    Draw curved bracket arcs outside the left pie chart to indicate
    Structure-Based vs Sequence-Based meta-groupings.
    """
    import math
    from matplotlib.patches import Arc

    meta_spans = {}
    for wedge, grp in zip(wedges, group_names):
        meta = META_GROUPS.get(grp, 'Other')
        t1, t2 = wedge.theta1, wedge.theta2
        if meta not in meta_spans:
            meta_spans[meta] = [t1, t2]
        else:
            meta_spans[meta][0] = min(meta_spans[meta][0], t1)
            meta_spans[meta][1] = max(meta_spans[meta][1], t2)

    tick_len = 0.10
    for meta_name, (ang_start, ang_end) in meta_spans.items():
        clr = META_PALETTE.get(meta_name, '#333')
        span = ang_end - ang_start
        if span < 0.5:
            continue

        arc = Arc((0, 0), 2 * arc_r, 2 * arc_r,
                  angle=0, theta1=ang_start, theta2=ang_end,
                  color=clr, lw=2.2, linestyle='-')
        ax.add_patch(arc)

        for ang_deg in [ang_start, ang_end]:
            rad = math.radians(ang_deg)
            ca, sa = math.cos(rad), math.sin(rad)
            ax.plot(
                [(arc_r - tick_len) * ca, (arc_r + tick_len) * ca],
                [(arc_r - tick_len) * sa, (arc_r + tick_len) * sa],
                color=clr, lw=2.2, solid_capstyle='round',
            )

        mid_deg = (ang_start + ang_end) / 2.0
        mid_rad = math.radians(mid_deg)
        tx = text_r * math.cos(mid_rad)
        ty = text_r * math.sin(mid_rad)
        ha = 'left' if math.cos(mid_rad) >= 0 else 'right'
        ax.text(tx, ty, meta_name, ha=ha, va='center',
                fontsize=10, fontweight='bold', color=clr,
                fontstyle='italic')


def plot_feature_importance_pies(imp_df, out_dir, model_type, pdb_id=None):
    """
    Produce TWO separate figures so each pie chart is full-size:

      Figure 1  -  Feature Categories pie  (sub-categories + meta-group arcs)
      Figure 2  -  Individual Features pie (top-20, legend text coloured by
                    Structure-Based vs Sequence-Based)

    Negative importances are clipped to zero and re-normalised to 100%.
    """
    if imp_df is None or imp_df.empty:
        return

    imp_df = imp_df.copy()

    imp_df['Imp (%)'] = imp_df['Imp (%)'].clip(lower=0)
    total = imp_df['Imp (%)'].sum()
    if total > 0:
        imp_df['Imp (%)'] = imp_df['Imp (%)'] / total * 100.0

    imp_df['Group'] = imp_df['Feature'].apply(_classify_feature)

    group_imp = (imp_df.groupby('Group')['Imp (%)']
                 .sum().reindex(GROUP_ORDER).fillna(0))
    group_imp = group_imp[group_imp > 0]
    palette1  = [GROUP_PALETTE[GROUP_ORDER.index(g)] for g in group_imp.index]
    g_labels  = [g.replace('\n', ' ') for g in group_imp.index]

    meta_totals = {}
    for grp, val in group_imp.items():
        meta = META_GROUPS.get(grp, 'Other')
        meta_totals[meta] = meta_totals.get(meta, 0.0) + val

    top_n     = 20
    imp_sort  = imp_df.sort_values('Imp (%)', ascending=False)
    top_df    = imp_sort.head(top_n)
    other_sum = imp_sort.iloc[top_n:]['Imp (%)'].sum()
    labels2   = list(top_df['Feature'])
    sizes2    = list(top_df['Imp (%)'])
    groups2   = list(top_df['Group'])
    if other_sum > 0.01:
        labels2.append(f'Others ({len(imp_sort) - top_n} feats)')
        sizes2.append(other_sum)
        groups2.append(None)
    cmap2   = plt.colormaps['tab20b'].resampled(len(labels2))
    colors2 = [cmap2(i) for i in range(len(labels2))]

    title_suffix = (f'  |  Structure: {pdb_id}' if pdb_id else '')
    prefix       = f'{pdb_id}_' if pdb_id else ''

    # FIGURE 1 - Feature Categories
    fig1 = plt.figure(figsize=(13, 14))
    gs1  = fig1.add_gridspec(2, 1, height_ratios=[5, 1.0],
                             top=0.93, bottom=0.02, left=0.02, right=0.98,
                             hspace=0.04)
    ax_pie1 = fig1.add_subplot(gs1[0])
    ax_leg1 = fig1.add_subplot(gs1[1])
    ax_leg1.axis('off')

    fig1.suptitle(
        f'Feature Importance \u2014 {model_type.upper()}{title_suffix}\n'
        f'Feature Categories',
        fontsize=17, fontweight='bold')

    wedges1, _ = ax_pie1.pie(
        group_imp.values, colors=palette1, startangle=140,
        explode=[0.03] * len(group_imp),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        labels=None, autopct=None,
    )
    _add_pie_leaders(ax_pie1, wedges1, g_labels, list(group_imp.values),
                     threshold=3.0, inner_r=0.65, outer_r=1.12, text_r=1.28)
    ax_pie1.set_xlim(-2.10, 2.10)
    ax_pie1.set_ylim(-2.10, 2.10)

    _draw_meta_brackets(ax_pie1, wedges1, list(group_imp.index))

    phys_val = meta_totals.get('Structure-Based', 0.0)
    seq_val  = meta_totals.get('Sequence-Based', 0.0)

    leg_handles = [plt.Rectangle((0, 0), 1, 1, fc=palette1[i], ec='white', lw=1.2)
                   for i in range(len(g_labels))]
    leg_labels  = [f'{lbl}  ({pct:.1f}%)'
                   for lbl, pct in zip(g_labels, group_imp.values)]

    leg_handles.append(plt.Rectangle((0, 0), 1, 1,
                       fc=META_PALETTE['Structure-Based'], ec='white', lw=1.2))
    leg_labels.append(f'Structure-Based: {phys_val:.1f}%')
    leg_handles.append(plt.Rectangle((0, 0), 1, 1,
                       fc=META_PALETTE['Sequence-Based'], ec='white', lw=1.2))
    leg_labels.append(f'Sequence-Based: {seq_val:.1f}%')

    ax_leg1.legend(
        leg_handles, leg_labels,
        loc='upper center', ncol=3,
        fontsize=10.5, frameon=True, framealpha=0.93,
        title='Feature Sub-Category', title_fontsize=12,
        handlelength=1.8, handleheight=1.5, labelspacing=0.7,
        columnspacing=2.0,
    )

    fname1 = f'{prefix}{model_type.lower()}_importance_categories.png'
    fig1.savefig(os.path.join(out_dir, fname1), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'  [PIE] Category importance  -> {fname1}')

    # FIGURE 2 - Individual Features (Top 20)
    fig2 = plt.figure(figsize=(13, 14))
    gs2  = fig2.add_gridspec(2, 1, height_ratios=[5, 1.3],
                             top=0.93, bottom=0.02, left=0.02, right=0.98,
                             hspace=0.04)
    ax_pie2 = fig2.add_subplot(gs2[0])
    ax_leg2 = fig2.add_subplot(gs2[1])
    ax_leg2.axis('off')

    fig2.suptitle(
        f'Feature Importance \u2014 {model_type.upper()}{title_suffix}\n'
        f'Individual Features (Top {top_n})',
        fontsize=17, fontweight='bold')

    wedges2, _ = ax_pie2.pie(
        sizes2, colors=colors2, startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
        labels=None, autopct=None,
    )
    _add_pie_leaders(ax_pie2, wedges2, labels2, sizes2,
                     threshold=3.5, inner_r=0.65, outer_r=1.12, text_r=1.28)
    ax_pie2.set_xlim(-1.55, 1.55)
    ax_pie2.set_ylim(-1.55, 1.55)

    leg = ax_leg2.legend(
        wedges2, labels2,
        loc='upper center', ncol=4,
        fontsize=10, frameon=True, framealpha=0.93,
        title=f'Top {top_n} Features', title_fontsize=12,
        handlelength=1.3, handleheight=1.3, labelspacing=0.6,
        columnspacing=1.5,
    )
    for txt_obj, grp in zip(leg.get_texts(), groups2):
        meta = META_GROUPS.get(grp, 'Other') if grp else 'Other'
        clr  = META_PALETTE.get(meta, '#333')
        txt_obj.set_color(clr)
        txt_obj.set_fontweight('bold')

    phys_clr = META_PALETTE['Structure-Based']
    seq_clr  = META_PALETTE['Sequence-Based']
    ax_leg2.text(0.5, 0.02, ' ', transform=ax_leg2.transAxes,
                 ha='center', va='bottom', fontsize=11)
    ax_leg2.annotate('Legend text colour:', xy=(0.28, 0.02),
                     xycoords='axes fraction', fontsize=10, color='#555',
                     ha='center', va='bottom', fontstyle='italic')
    ax_leg2.annotate('Structure-Based', xy=(0.44, 0.02),
                     xycoords='axes fraction', fontsize=11,
                     fontweight='bold', color=phys_clr, ha='center', va='bottom')
    ax_leg2.annotate('|', xy=(0.575, 0.02), xycoords='axes fraction',
                     fontsize=11, color='#666', ha='center', va='bottom')
    ax_leg2.annotate('Sequence-Based', xy=(0.66, 0.02), xycoords='axes fraction',
                     fontsize=11, fontweight='bold', color=seq_clr,
                     ha='center', va='bottom')

    fname2 = f'{prefix}{model_type.lower()}_importance_individual.png'
    fig2.savefig(os.path.join(out_dir, fname2), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'  [PIE] Individual importance -> {fname2}')

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION & GROUND TRUTH COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def predict_and_compare(model_dir, features_df, pdb_id):
    try:
        model = joblib.load(os.path.join(model_dir, "pka_model.pkl"))
        model_features = joblib.load(os.path.join(model_dir, "model_features.pkl"))
        df = features_df.copy()

        X = pd.get_dummies(df, columns=['resi_type']).reindex(columns=model_features, fill_value=0)
        df['Pred_Shift'] = np.round(model.predict(X), 3)
        df['Pred_pKa'] = np.round(df['pKa0'] + df['Pred_Shift'], 3)

        has_gt = False
        target_df = None

        holdout_csv = os.path.join(model_dir, "holdout_analysis.csv")
        if os.path.exists(holdout_csv):
            h_df = pd.read_csv(holdout_csv)
            target_df = h_df[h_df['PDB'].str.upper() == pdb_id.upper()]

        master_pickle = "../../ml_dataset.pkl"
        if (target_df is None or target_df.empty) and os.path.exists(master_pickle):
            m_df = pd.read_pickle(master_pickle)
            target_df = m_df[m_df['PDB'].str.upper() == pdb_id.upper()]

        if target_df is None or target_df.empty:
            for item in os.listdir(model_dir):
                test_pred_csv = os.path.join(model_dir, item, "test_predictions.csv")
                if item.startswith("test_") and os.path.exists(test_pred_csv):
                    t_df = pd.read_csv(test_pred_csv)
                    match = t_df[t_df['PDB'].str.upper() == pdb_id.upper()]
                    if not match.empty:
                        target_df = match; break

        if target_df is not None and not target_df.empty:
            if 'resi+chainId+resid' in target_df.columns and 'Target_pKa_shift' in target_df.columns:
                df = df.merge(
                    target_df[['resi+chainId+resid', 'Target_pKa_shift']].drop_duplicates(),
                    on='resi+chainId+resid', how='left')
                df['MCCE_pKa'] = np.round(df['pKa0'] + df['Target_pKa_shift'], 3)
                df['Error'] = np.round(df['Pred_pKa'] - df['MCCE_pKa'], 3)
                has_gt = True

        return df, model_features, model, has_gt
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}"); return None, None, None, False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mcce2ml_pKa: Predict pKa for a single PDB structure.")
    parser.add_argument("pdb_file", help="Input PDB structure")
    parser.add_argument("-m", "--model", default="MCCE_ML-models/model_lgbm")
    parser.add_argument("--all", action="store_true", help="Run across all trained models")
    parser.add_argument("-r", "--run_step1", action="store_true", help="Force MCCE step1.py")
    args = parser.parse_args()

    pdb_id = os.path.basename(args.pdb_file).replace('.pdb', '').upper()
    if args.run_step1 or not os.path.exists("acc.res"):
        if not run_mcce_step1(args.pdb_file): return
    sasa_lookup = parse_acc_res()

    print(f"\n[INFO] Extracting structural + sequence features for {pdb_id}...")
    base_features = generate_features_for_predict(args.pdb_file, sasa_lookup)
    if base_features.empty:
        print("[ERROR] No titratable residues found."); return

    out_dir = os.path.abspath(f"PREDICTIONS_{pdb_id}")
    os.makedirs(out_dir, exist_ok=True)

    model_root = "MCCE_ML-models"
    if args.all:
        models = [os.path.join(model_root, d) for d in os.listdir(model_root)
                  if os.path.isdir(os.path.join(model_root, d))]
    else:
        models = [args.model]

    for m_dir in models:
        m_name = os.path.basename(m_dir).replace('model_', '').upper()
        print(f"\n{'='*95}\nMODEL: {m_name} | TARGET STRUCTURE: {pdb_id}\n{'='*95}")
        df, feats, model_obj, matched = predict_and_compare(m_dir, base_features, pdb_id)
        if df is None: continue

        cols = ['resi+chainId+resid', 'pKa0', 'Pred_pKa', 'Pred_Shift']
        if matched:
            cols += ['MCCE_pKa', 'Error']
            print(f"[MATCH FOUND] Historical MCCE Ground Truth integrated into report.\n")

        print(df[cols].to_string(index=False))

        if matched:
            valid = df.dropna(subset=['Error'])
            err = valid['Error'].abs()
            print(f"\nINDIVIDUAL STRUCTURE PERFORMANCE ({m_name}):")
            print(f"MAE: {err.mean():.3f} | "
                  f"±1.0: {(err<=1.0).mean()*100:.1f}% | "
                  f"±2.0: {(err<=2.0).mean()*100:.1f}% | "
                  f"±3.0: {(err<=3.0).mean()*100:.1f}%")

        # Feature importance
        inner = model_obj.named_steps.get(m_name.lower()) if hasattr(model_obj, 'named_steps') else model_obj
        if inner is not None and hasattr(inner, 'feature_importances_'):
            imp = inner.feature_importances_
            imp_df = pd.DataFrame({
                'Feature': feats, 'Imp (%)': (imp / np.sum(imp)) * 100
            }).sort_values('Imp (%)', ascending=False)
            print(f"\nTOP PHYSICS DRIVERS ({m_name}):\n{imp_df.head(10).to_string(index=False)}")
            plot_feature_importance_pies(imp_df, out_dir, m_name, pdb_id=pdb_id)

        df.to_csv(os.path.join(out_dir, f"{pdb_id}_{m_name.lower()}_inference.csv"), index=False)

    print(f"\n{'='*85}\nINFERENCE COMPLETE: Outputs saved in '{out_dir}/'\n{'='*85}")


if __name__ == "__main__":
    main()
