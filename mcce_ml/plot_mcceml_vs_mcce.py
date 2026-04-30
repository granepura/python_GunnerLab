#!/usr/bin/env python3
"""
Plot MCCE-ML predicted pKa vs MCCE pK.out values across all SEQ*_REP* directories.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = "/data/home/granepura/Marwell/mcce4"

RESTYPE_COLORS = {
    "ASP": "#e63946",
    "GLU": "#f4a261",
    "HIS": "#2a9d8f",
    "LYS": "#264653",
    "ARG": "#457b9d",
    "TYR": "#8338ec",
}


def parse_pk_out(pk_path):
    """Parse pK.out, return dict of {residue_id: pKa} for numeric values only."""
    results = {}
    with open(pk_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("pH"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_id = parts[0]
            pka_str = parts[1]

            if pka_str.startswith(">"):
                pka = 14.0
            elif pka_str.startswith("<"):
                pka = 0.0
            else:
                try:
                    pka = float(pka_str)
                except ValueError:
                    continue

            clean_id = raw_id.replace("+", "").replace("-", "").rstrip("_")
            results[clean_id] = pka
    return results


def parse_inference_csv(csv_path):
    """Parse MCCE-ML inference CSV, return dict of {residue_id: Pred_pKa}."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["resi+chainId+resid"], df["Pred_pKa"]))


def collect_all_data():
    """Walk all SEQ*_REP* dirs and collect matched pairs."""
    rows = []
    dirs = sorted(glob.glob(os.path.join(BASE, "SEQ*_REP*")))

    for d in dirs:
        name = os.path.basename(d)
        pk_path = os.path.join(d, "pK.out")
        inf_path = os.path.join(d, f"PREDICTIONS_{name}", f"{name}_lgbm_inference.csv")

        if not os.path.isfile(pk_path) or not os.path.isfile(inf_path):
            continue

        mcce_pkas = parse_pk_out(pk_path)
        ml_pkas = parse_inference_csv(inf_path)

        seq_label = name.split("_REP")[0]

        for resid, mcce_val in mcce_pkas.items():
            if resid in ml_pkas:
                restype = resid[:3]
                rows.append({
                    "directory": name,
                    "sequence": seq_label,
                    "residue": resid,
                    "restype": restype,
                    "MCCE_pKa": mcce_val,
                    "ML_pKa": ml_pkas[resid],
                })

    return pd.DataFrame(rows)


def plot_combined(df, out_path):
    """Single combined scatter: ML vs MCCE, colored by residue type."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for restype in sorted(RESTYPE_COLORS.keys()):
        sub = df[df["restype"] == restype]
        if sub.empty:
            continue
        ax.scatter(sub["MCCE_pKa"], sub["ML_pKa"],
                   c=RESTYPE_COLORS[restype], label=restype,
                   alpha=0.5, s=20, edgecolors="none")

    lo = min(df["MCCE_pKa"].min(), df["ML_pKa"].min()) - 0.5
    hi = max(df["MCCE_pKa"].max(), df["ML_pKa"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)

    envelope_offsets = [0.5, 1.0, 2.0]
    envelope_styles = [("gray", 0.8, 0.5, ":"),
                       ("gray", 0.6, 0.8, "-."),
                       ("gray", 0.4, 1.0, "--")]
    for offset, (color, alpha, lw_env, ls) in zip(envelope_offsets, envelope_styles):
        ax.plot([lo, hi], [lo + offset, hi + offset], color=color,
                ls=ls, lw=lw_env, alpha=alpha)
        ax.plot([lo, hi], [lo - offset, hi - offset], color=color,
                ls=ls, lw=lw_env, alpha=alpha,
                label=f"±{offset:.1f}" if offset == envelope_offsets[0] else f"±{offset:.1f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    mae = mean_absolute_error(df["MCCE_pKa"], df["ML_pKa"])
    rmse = np.sqrt(mean_squared_error(df["MCCE_pKa"], df["ML_pKa"]))
    r2 = r2_score(df["MCCE_pKa"], df["ML_pKa"])
    n = len(df)

    residuals = np.abs(df["ML_pKa"].values - df["MCCE_pKa"].values)
    within_05 = int(np.sum(residuals <= 0.5))
    within_1 = int(np.sum(residuals <= 1.0))
    within_2 = int(np.sum(residuals <= 2.0))

    ax.set_xlabel("MCCE pKa (from pK.out)", fontsize=14)
    ax.set_ylabel("MCCE-ML Predicted pKa", fontsize=14)
    ax.set_title("MCCE-ML Predicted pKa vs MCCE pKa\n(All Sequences & Replicates)", fontsize=16)

    stats_text = (f"N = {n}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.3f}\n"
                  f"±0.5: {within_05} ({100*within_05/n:.1f}%)\n"
                  f"±1.0: {within_1} ({100*within_1/n:.1f}%)\n"
                  f"±2.0: {within_2} ({100*within_2/n:.1f}%)")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    ax.legend(title="Residue Type", fontsize=11, title_fontsize=12,
              loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot: {out_path}")


def plot_per_sequence(df, out_dir):
    """One subplot per sequence (averaging over replicates shown as individual points)."""
    sequences = sorted(df["sequence"].unique(),
                       key=lambda s: int(re.search(r"\d+", s).group()))
    n_seq = len(sequences)
    ncols = 4
    nrows = (n_seq + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, seq in enumerate(sequences):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["sequence"] == seq]

        for restype in sorted(RESTYPE_COLORS.keys()):
            rs = sub[sub["restype"] == restype]
            if rs.empty:
                continue
            ax.scatter(rs["MCCE_pKa"], rs["ML_pKa"],
                       c=RESTYPE_COLORS[restype], label=restype,
                       alpha=0.5, s=15, edgecolors="none")

        lo = min(sub["MCCE_pKa"].min(), sub["ML_pKa"].min()) - 0.5
        hi = max(sub["MCCE_pKa"].max(), sub["ML_pKa"].max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        mae = mean_absolute_error(sub["MCCE_pKa"], sub["ML_pKa"])
        r2 = r2_score(sub["MCCE_pKa"], sub["ML_pKa"])
        ax.set_title(f"{seq}  (MAE={mae:.2f}, R²={r2:.2f})", fontsize=11)
        ax.set_xlabel("MCCE pKa", fontsize=9)
        ax.set_ylabel("ML pKa", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n_seq, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = [], []
    for restype in sorted(RESTYPE_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=RESTYPE_COLORS[restype],
                                  markersize=8))
        labels.append(restype)
    fig.legend(handles, labels, loc="upper center", ncol=6,
               fontsize=11, framealpha=0.9, title="Residue Type",
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("MCCE-ML vs MCCE pKa — Per Sequence", fontsize=18, y=1.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "mcceml_vs_mcce_per_sequence.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-sequence plot: {out_path}")


def plot_per_restype(df, out_dir):
    """One subplot per residue type."""
    restypes = sorted(df["restype"].unique())
    n = len(restypes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             squeeze=False)

    for idx, restype in enumerate(restypes):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["restype"] == restype]

        ax.scatter(sub["MCCE_pKa"], sub["ML_pKa"],
                   c=RESTYPE_COLORS.get(restype, "gray"),
                   alpha=0.4, s=20, edgecolors="none")

        lo = min(sub["MCCE_pKa"].min(), sub["ML_pKa"].min()) - 0.5
        hi = max(sub["MCCE_pKa"].max(), sub["ML_pKa"].max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        mae = mean_absolute_error(sub["MCCE_pKa"], sub["ML_pKa"])
        rmse = np.sqrt(mean_squared_error(sub["MCCE_pKa"], sub["ML_pKa"]))
        r2 = r2_score(sub["MCCE_pKa"], sub["ML_pKa"])
        n_pts = len(sub)

        ax.set_title(f"{restype}  (N={n_pts})", fontsize=14, fontweight="bold")
        ax.set_xlabel("MCCE pKa", fontsize=11)
        ax.set_ylabel("ML pKa", fontsize=11)

        stats = f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nR²={r2:.3f}"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        ax.grid(True, alpha=0.3)

    for idx in range(len(restypes), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MCCE-ML vs MCCE pKa — Per Residue Type", fontsize=18)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "mcceml_vs_mcce_per_restype.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-residue-type plot: {out_path}")


if __name__ == "__main__":
    print("Collecting data from all SEQ*_REP* directories...")
    df = collect_all_data()
    print(f"Found {len(df)} matched residue pairs across {df['directory'].nunique()} directories")
    print(f"Sequences: {sorted(df['sequence'].unique())}")
    print(f"Residue types: {sorted(df['restype'].unique())}")

    out_dir = os.path.join(BASE, "MCCEML_vs_MCCE_plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_combined(df, os.path.join(out_dir, "mcceml_vs_mcce_combined.png"))
    plot_per_sequence(df, out_dir)
    plot_per_restype(df, out_dir)

    summary_path = os.path.join(out_dir, "summary_stats.csv")
    stats = []
    for seq in sorted(df["sequence"].unique()):
        sub = df[df["sequence"] == seq]
        stats.append({
            "Sequence": seq,
            "N_residues": len(sub),
            "MAE": round(mean_absolute_error(sub["MCCE_pKa"], sub["ML_pKa"]), 3),
            "RMSE": round(np.sqrt(mean_squared_error(sub["MCCE_pKa"], sub["ML_pKa"])), 3),
            "R2": round(r2_score(sub["MCCE_pKa"], sub["ML_pKa"]), 3),
        })
    pd.DataFrame(stats).to_csv(summary_path, index=False)
    print(f"Saved summary stats: {summary_path}")
    print("Done.")
