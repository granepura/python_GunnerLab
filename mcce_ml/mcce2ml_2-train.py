#!/usr/bin/env python3
"""
mcce2ml_2-train.py
==================
Combined training + cross-model comparison for MCCE-ML.

Modes
-----
  Training (default):
      python mcce2ml_2-train.py -m lgbm
      python mcce2ml_2-train.py -m rf -t myoglobin_ml_dataset.pkl
      python mcce2ml_2-train.py --all
      python mcce2ml_2-train.py --all --skip_compare

  Comparison only (no training):
      python mcce2ml_2-train.py --compare
      python mcce2ml_2-train.py --compare --base_dir MCCE_ML-models --out_dir comparison_plots
"""

import os
import sys
import math
import argparse
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state
from sklearn.inspection import permutation_importance

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError:
    LGBMRegressor = None

warnings.filterwarnings("ignore")


# ===============================================================================
# CONSTANTS
# ===============================================================================

COLORS = {
    'LGBM':     '#1b5e20',
    'RF':       '#b71c1c',
    'SVR':      '#4a148c',
    'MLP':      '#01579b',
    'BAYESIAN': '#fbc02d',
    'ELASTIC':  '#00695c',
    'KNN':      '#e65100',
}
MODEL_ORDER = ['LGBM', 'RF', 'SVR', 'MLP', 'BAYESIAN', 'ELASTIC', 'KNN']

# Feature group definitions (mirror README taxonomy exactly)
# README: "44 features: 11 structural + 33 sequence-derived,
#           plus a one-hot encoding of residue type"
_STRUCTURAL_CORE = {
    'pKa0', 'SASA_rel', 'Polarity_Ratio', 'Backbone_Density',
    'Total_Packing', 'HBond_Potential', 'Net_Charge_6A', 'EPI',
    'LD1_count', 'LD2_count', 'LD3_count',
}

# Six sub-categories for the category pie chart.
# ORDER matters: Structure-Based groups first, then Sequence-Based,
# so the bracket arcs span contiguous slices without overlapping.
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
# Structure-Based: Structural Features (11), Residue Type one-hots
# Sequence-Based:  AAC (20), Physicochemical (8), Transition (4), SeqLen (1)
META_GROUPS = {
    'Structural\nFeatures (11)':       'Structure-Based',
    'Residue Type\n(One-Hot)':         'Structure-Based',
    'Amino Acid\nComposition (20)':    'Sequence-Based',
    'Physicochemical\nProperties (8)': 'Sequence-Based',
    'Transition\nFeatures (4)':        'Sequence-Based',
    'Sequence\nLength (1)':            'Sequence-Based',
}
META_PALETTE = {'Structure-Based': '#1565C0', 'Sequence-Based': '#C62828'}


# ===============================================================================
# UTILITIES
# ===============================================================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def get_model(model_type, lr, depth):
    if model_type == 'rf':
        return RandomForestRegressor(n_estimators=300, max_depth=depth,
                                     min_samples_leaf=3, random_state=42)
    elif model_type == 'lgbm' and LGBMRegressor is not None:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lgbm', LGBMRegressor(n_estimators=1000, learning_rate=lr,
                                   num_leaves=31, max_depth=depth,
                                   min_child_samples=20, random_state=42,
                                   verbose=-1, metric='rmse')),
        ])
    elif model_type == 'svr':
        return Pipeline([('scaler', StandardScaler()),
                         ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1))])
    elif model_type == 'bayesian':
        return BayesianRidge()
    elif model_type == 'mlp':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01,
                                 learning_rate_init=lr, max_iter=1000,
                                 random_state=42, early_stopping=True)),
        ])
    elif model_type == 'elastic':
        return Pipeline([('scaler', StandardScaler()),
                         ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5,
                                                random_state=42))])
    elif model_type == 'knn':
        return Pipeline([('scaler', StandardScaler()),
                         ('knn', KNeighborsRegressor(n_neighbors=15,
                                                     weights='uniform'))])
    return None


def get_full_stats(y, p):
    y, p = np.asarray(y), np.asarray(p)
    err = np.abs(y - p)
    return {
        "MAE":  mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2":   r2_score(y, p),
        "pm1":  np.mean(err <= 1.0) * 100,
        "pm2":  np.mean(err <= 2.0) * 100,
        "pm3":  np.mean(err <= 3.0) * 100,
        "N":    len(y),
    }


def get_dist_string(df):
    """Residue composition string.  Works with one-hot, string, or id columns."""
    if 'resi_type' in df.columns:
        counts = df['resi_type'].value_counts().to_dict()
    else:
        res_cols = [c for c in df.columns if c.startswith('resi_type_')]
        if res_cols:
            counts = {c.replace('resi_type_', ''): int(df[c].sum())
                      for c in res_cols}
        else:
            counts = df['resi+chainId+resid'].str[:3].value_counts().to_dict()
    return "  ".join(
        f"{k}:{v}" for k, v in
        sorted(counts.items(), key=lambda x: x[1], reverse=True) if v > 0
    )


def _derive_test_name(test_data_path):
    basename = os.path.splitext(os.path.basename(test_data_path))[0]
    if basename == "ml_dataset":
        parent = os.path.basename(
            os.path.dirname(os.path.abspath(test_data_path)))
        return parent if parent else basename
    if basename.endswith("_ml_dataset"):
        return basename[:-len("_ml_dataset")]
    return basename


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


# ===============================================================================
# TRAINING PLOTS
# ===============================================================================

def plot_combined_cv_loss(all_cv_evals, model_type, output_dir):
    if not all_cv_evals:
        return
    plt.figure(figsize=(12, 7))
    colors = plt.colormaps['tab10'].resampled(len(all_cv_evals))
    for i, eval_data in enumerate(all_cv_evals):
        curve = (eval_data['valid_0']['rmse']
                 if isinstance(eval_data, dict) else eval_data)
        plt.plot(curve, color=colors(i), label=f'Fold {i+1} Val Loss')
    plt.xlabel('Iterations', fontsize=16); plt.ylabel('RMSE/Loss', fontsize=16)
    plt.legend(ncol=2, fontsize=12)
    plt.title(f'Combined CV Loss Curves (Training Pool): {model_type.upper()}',
              fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', labelsize=13)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss_curve_cv_combined.png"), dpi=300)
    plt.close()


def create_dashboard(y_t, p_t, y_v, p_v, df_t, df_v,
                     model_name, output_dir, mode="shift"):
    sns.set_style("whitegrid")
    lims  = (-10, 10) if mode == "shift" else (0, 14)
    label = "pKa Shift" if mode == "shift" else "Absolute pKa"
    fname = ("performance_correlation.png" if mode == "shift"
             else "performance_correlation_pKa.png")

    s_t, s_v = get_full_stats(y_t, p_t), get_full_stats(y_v, p_v)
    stats_text = (
        f"DETAILED {label.upper()} PERFORMANCE\n{'='*55}\n"
        f"TRAIN (80%): {len(y_t)} res\n"
        f"MAE: {s_t['MAE']:.2f} | RMSE: {s_t['RMSE']:.2f} | R2: {s_t['R2']:.2f}\n"
        f"+-1.0: {s_t['pm1']:.1f}% | +-2.0: {s_t['pm2']:.1f}% | +-3.0: {s_t['pm3']:.1f}%\n"
        f"DIST: {get_dist_string(df_t)}\n{'-'*45}\n"
        f"VAL (20%): {len(y_v)} res\n"
        f"MAE: {s_v['MAE']:.2f} | RMSE: {s_v['RMSE']:.2f} | R2: {s_v['R2']:.2f}\n"
        f"+-1.0: {s_v['pm1']:.1f}% | +-2.0: {s_v['pm2']:.1f}% | +-3.0: {s_v['pm3']:.1f}%\n"
        f"DIST: {get_dist_string(df_v)}\n{'='*55}"
    )

    fig      = plt.figure(figsize=(13, 17))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[5, 1.5], hspace=0.18)
    ax       = fig.add_subplot(gs_outer[0])
    ax_text  = fig.add_subplot(gs_outer[1])
    ax_text.axis("off")

    x_range = np.linspace(lims[0], lims[1], 100)
    ax.plot(lims, lims, color='black', lw=2, label='Identity', zorder=1)
    for i, (ls, alpha) in enumerate([('--', 0.4), (':', 0.25), ('-.', 0.15)], 1):
        ax.plot(x_range, x_range + i, color='gray', linestyle=ls, alpha=alpha, zorder=1)
        ax.plot(x_range, x_range - i, color='gray', linestyle=ls, alpha=alpha, zorder=1)

    ax.scatter(y_t, p_t, alpha=0.2, color='#2ecc71', s=20,
               label='Train Pool (80%)', zorder=3)
    ax.scatter(y_v, p_v, alpha=0.5, color='#e74c3c', s=50,
               edgecolor='white', lw=0.5, label='Val Holdout (20%)', zorder=4)

    ax.set_title(f'{label} Correlation Dashboard: {model_name.upper()}', fontsize=24, fontweight='bold')
    ax.set_xlabel(f'MCCE {label}', fontsize=18)
    ax.set_ylabel(f'Predicted {label}', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(loc='lower right', fontsize=14)

    txt = ax_text.text(0.5, 0.8, stats_text, transform=ax_text.transAxes,
                       family='monospace', fontsize=11,
                       verticalalignment='top', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='white',
                                 alpha=0.9, edgecolor='gray'))

    # Scale font so the textbox fills the plot width
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox  = ax.get_window_extent(renderer=renderer)
    txt_bbox = txt.get_window_extent(renderer=renderer)
    if txt_bbox.width > 0:
        scale = min(ax_bbox.width * 0.98 / txt_bbox.width, 1.6)
        txt.set_fontsize(11 * scale)

    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()


# ===============================================================================
# EXTERNAL TEST EVALUATION
# ===============================================================================

def create_test_dashboard(y_test, p_test, df_test,
                          model_name, test_name, test_dir, mode="shift"):
    sns.set_style("whitegrid")
    lims  = (-10, 10) if mode == "shift" else (0, 14)
    label = "pKa Shift" if mode == "shift" else "Absolute pKa"
    fname = ("test_correlation_shift.png" if mode == "shift"
             else "test_correlation_pKa.png")

    s = get_full_stats(y_test, p_test)
    stats_text = (
        f"EXTERNAL TEST {label.upper()} PERFORMANCE\n{'='*55}\n"
        f"TEST SET: {len(y_test)} residues ({test_name})\n"
        f"MAE: {s['MAE']:.2f} | RMSE: {s['RMSE']:.2f} | R2: {s['R2']:.2f}\n"
        f"+-1.0: {s['pm1']:.1f}% | +-2.0: {s['pm2']:.1f}% | +-3.0: {s['pm3']:.1f}%\n"
        f"DIST: {get_dist_string(df_test)}\n{'='*55}"
    )

    fig      = plt.figure(figsize=(13, 17))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[5, 1.2], hspace=0.18)
    ax       = fig.add_subplot(gs_outer[0])
    ax_text  = fig.add_subplot(gs_outer[1])
    ax_text.axis("off")

    x_range = np.linspace(lims[0], lims[1], 100)
    ax.plot(lims, lims, color='black', lw=2, label='Identity', zorder=1)
    for i, (ls, alpha) in enumerate([('--', 0.4), (':', 0.25), ('-.', 0.15)], 1):
        ax.plot(x_range, x_range + i, color='gray', linestyle=ls, alpha=alpha, zorder=1)
        ax.plot(x_range, x_range - i, color='gray', linestyle=ls, alpha=alpha, zorder=1)

    ax.scatter(y_test, p_test, alpha=0.6, color='#3498db', s=50,
               edgecolor='white', lw=0.5, label=f'Test: {test_name}', zorder=4)
    ax.set_xlabel(f'MCCE {label}', fontsize=18)
    ax.set_ylabel(f'Predicted {label}', fontsize=18)
    ax.set_title(f'External Test {label}: {model_name.upper()} on {test_name}',
                 fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(loc='lower right', fontsize=14)

    txt = ax_text.text(0.5, 0.97, stats_text, transform=ax_text.transAxes,
                       family='monospace', fontsize=11,
                       verticalalignment='top', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='white',
                                 alpha=0.9, edgecolor='gray'))

    # Scale font so the textbox fills the plot width
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox  = ax.get_window_extent(renderer=renderer)
    txt_bbox = txt.get_window_extent(renderer=renderer)
    if txt_bbox.width > 0:
        scale = min(ax_bbox.width * 0.98 / txt_bbox.width, 1.6)
        txt.set_fontsize(11 * scale)

    plt.savefig(os.path.join(test_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_error_distribution(y_test, p_test, test_name, test_dir):
    errors = p_test - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=40, color='#3498db', edgecolor='white', alpha=0.8)
    plt.axvline(0, color='black', lw=1.5, linestyle='--')
    plt.axvline(np.mean(errors), color='red', lw=1.5, linestyle='-',
                label=f'Mean Bias: {np.mean(errors):.2f}')
    plt.xlabel('Signed Error (Predicted - Experimental)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title(f'Error Distribution on External Test Set: {test_name}',
              fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(fontsize=13); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(test_dir, 'test_error_distribution.png'), dpi=300)
    plt.close()


def plot_test_per_residue(y_test, p_test, df_test, test_dir):
    res_cols = [c for c in df_test.columns if c.startswith('resi_type_')]
    abs_err  = np.abs(p_test - y_test)
    records  = []
    for c in res_cols:
        rtype = c.replace('resi_type_', '')
        mask  = df_test[c].values == 1
        if mask.sum() > 0:
            for e in abs_err[mask]:
                records.append({'Residue Type': rtype, 'Absolute Error': e})
    if not records:
        return
    err_df = pd.DataFrame(records)
    order  = (err_df.groupby('Residue Type')['Absolute Error']
              .median().sort_values().index.tolist())
    plt.figure(figsize=(max(8, len(order) * 1.2), 6))
    sns.boxplot(data=err_df, x='Residue Type', y='Absolute Error',
                order=order, palette='Set2', fliersize=3)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='+-1.0 pKa')
    plt.axhline(2.0, color='gray', linestyle=':', alpha=0.4, label='+-2.0 pKa')
    plt.title('Per-Residue-Type Error on External Test Set',
              fontsize=18, fontweight='bold')
    plt.xlabel('Residue Type', fontsize=16)
    plt.ylabel('Absolute Error', fontsize=16)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(fontsize=13); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'test_per_residue_error.png'), dpi=300)
    plt.close()


def plot_test_per_protein(y_test, p_test, df_test, test_dir):
    pdbs    = df_test['PDB'].values
    abs_err = np.abs(p_test - y_test)
    pdb_stats = (pd.DataFrame({'PDB': pdbs, 'AbsErr': abs_err})
                 .groupby('PDB')
                 .agg(MAE=('AbsErr', 'mean'), Count=('AbsErr', 'size'))
                 .reset_index()
                 .sort_values('MAE', ascending=False))
    cmap   = plt.colormaps['RdYlGn_r'].resampled(len(pdb_stats))
    colors = [cmap(i / max(1, len(pdb_stats) - 1)) for i in range(len(pdb_stats))]
    plt.figure(figsize=(max(10, len(pdb_stats) * 0.5), 6))
    plt.bar(range(len(pdb_stats)), pdb_stats['MAE'].values,
            color=colors, edgecolor='white')
    plt.xticks(range(len(pdb_stats)), pdb_stats['PDB'].values,
               rotation=90, fontsize=9)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel('MAE (pKa Shift)', fontsize=16)
    plt.xlabel('PDB ID', fontsize=16)
    plt.title('Per-Protein MAE on External Test Set',
              fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'test_per_protein_mae.png'), dpi=300)
    plt.close()
    return pdb_stats


def run_test_evaluation(final_model, train_features, model_type,
                        model_output_dir, test_data_path):
    test_name = _derive_test_name(test_data_path)
    test_dir  = os.path.join(model_output_dir, f"test_{test_name}")
    os.makedirs(test_dir, exist_ok=True)

    log_path        = os.path.join(test_dir, "test_evaluation_log.txt")
    original_stdout = sys.stdout
    sys.stdout      = Logger(log_path)

    print(f"{'='*85}\nEXTERNAL TEST EVALUATION: {test_name}\n"
          f"Model: {model_type.upper()}\n"
          f"Source: {os.path.abspath(test_data_path)}\n{'='*85}\n")

    df_test_raw = pd.read_pickle(test_data_path)
    df_test     = pd.get_dummies(df_test_raw, columns=['resi_type'])
    y_test      = df_test['Target_pKa_shift']
    X_test      = df_test.drop(
        columns=['PDB', 'resi+chainId+resid', 'Target_pKa_shift'])

    missing = [f for f in train_features if f not in X_test.columns]
    extra   = [f for f in X_test.columns if f not in train_features]
    for f in missing:
        X_test[f] = 0
    X_test = X_test[train_features]
    if missing:
        print(f"[ALIGN] Added {len(missing)} zero-filled features: {missing}")
    if extra:
        print(f"[ALIGN] Dropped {len(extra)} features not in model: {extra}")

    print(f"Test samples: {len(y_test)} | Features: {X_test.shape[1]} | "
          f"PDBs: {df_test['PDB'].nunique()}")
    print(f"COMPOSITION: {get_dist_string(df_test)}\n")

    p_test_shift = final_model.predict(X_test)
    s = get_full_stats(y_test.values, p_test_shift)
    print(f"{'='*60}\nGLOBAL TEST METRICS (pKa Shift)\n{'='*60}")
    for k, v in [("MAE", s['MAE']), ("RMSE", s['RMSE']), ("R2", s['R2']),
                 ("+-1.0", s['pm1']), ("+-2.0", s['pm2']), ("+-3.0", s['pm3'])]:
        fmt = f"{v:.1f}%" if k.startswith("+") else f"{v:.3f}"
        print(f"  {k:<6}: {fmt}")
    print(f"  Bias  : {np.mean(p_test_shift - y_test.values):.3f}")

    has_pka0 = 'pKa0' in df_test.columns
    if has_pka0:
        y_pka = df_test['pKa0'].values + y_test.values
        p_pka = df_test['pKa0'].values + p_test_shift
        s_pka = get_full_stats(y_pka, p_pka)
        print(f"\n{'='*60}\nGLOBAL TEST METRICS (Absolute pKa)\n{'='*60}")
        for k, v in [("MAE", s_pka['MAE']), ("RMSE", s_pka['RMSE']),
                     ("R2", s_pka['R2']), ("+-1.0", s_pka['pm1']),
                     ("+-2.0", s_pka['pm2']), ("+-3.0", s_pka['pm3'])]:
            fmt = f"{v:.1f}%" if k.startswith("+") else f"{v:.3f}"
            print(f"  {k:<6}: {fmt}")

    res_cols     = [c for c in df_test.columns if c.startswith('resi_type_')]
    per_res_rows = []
    print(f"\n{'-'*80}\nPER-RESIDUE-TYPE BREAKDOWN\n{'-'*80}")
    print(f"{'Residue':<10} | {'N':<5} | {'MAE':<7} | {'RMSE':<7} | "
          f"{'R2':<7} | {'+-1.0':<7} | {'+-2.0':<7}")
    for c in sorted(res_cols):
        rtype = c.replace('resi_type_', '')
        mask  = df_test[c].values == 1
        n     = mask.sum()
        if n < 2:
            continue
        rs = get_full_stats(y_test.values[mask], p_test_shift[mask])
        print(f"{rtype:<10} | {n:<5} | {rs['MAE']:<7.3f} | {rs['RMSE']:<7.3f} | "
              f"{rs['R2']:<7.3f} | {rs['pm1']:<7.1f} | {rs['pm2']:<7.1f}")
        per_res_rows.append({'Residue': rtype, 'N': n, **rs})
    if per_res_rows:
        pd.DataFrame(per_res_rows).to_csv(
            os.path.join(test_dir, 'test_per_residue_stats.csv'), index=False)

    print(f"\n{'-'*80}\nPER-PROTEIN BREAKDOWN\n{'-'*80}")
    print(f"{'PDB':<8} | {'N':<5} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7}")
    per_pdb_rows = []
    for pdb in sorted(df_test['PDB'].unique()):
        mask = df_test['PDB'].values == pdb
        n    = mask.sum()
        if n < 2:
            continue
        ps = get_full_stats(y_test.values[mask], p_test_shift[mask])
        print(f"{pdb:<8} | {n:<5} | {ps['MAE']:<7.3f} | "
              f"{ps['RMSE']:<7.3f} | {ps['R2']:<7.3f}")
        per_pdb_rows.append({'PDB': pdb, 'N': n, **ps})
    if per_pdb_rows:
        pd.DataFrame(per_pdb_rows).to_csv(
            os.path.join(test_dir, 'test_per_protein_stats.csv'), index=False)

    create_test_dashboard(y_test.values, p_test_shift, df_test,
                          model_type, test_name, test_dir, mode="shift")
    if has_pka0:
        create_test_dashboard(y_pka, p_pka, df_test,
                              model_type, test_name, test_dir, mode="pka")
    plot_test_error_distribution(y_test.values, p_test_shift, test_name, test_dir)
    plot_test_per_residue(y_test.values, p_test_shift, df_test, test_dir)
    plot_test_per_protein(y_test.values, p_test_shift, df_test, test_dir)

    out_df = df_test.copy()
    out_df['Predicted_pKa_shift'] = p_test_shift
    if has_pka0:
        out_df['Predicted_pKa'] = df_test['pKa0'].values + p_test_shift
    out_df.to_csv(os.path.join(test_dir, 'test_predictions.csv'), index=False)

    artifacts = sorted(os.listdir(test_dir))
    print(f"\n{'='*85}\nTEST ARTIFACTS ({test_name})\n{'='*85}")
    for i, f in enumerate(artifacts, 1):
        print(f"  {i}. {f}")
    print(f"\nAll saved to: {os.path.abspath(test_dir)}")

    sys.stdout = original_stdout
    print(f"  [TEST] {model_type.upper()} on {test_name}: "
          f"MAE={s['MAE']:.3f}  R2={s['R2']:.3f}")
    return s


# ===============================================================================
# FEATURE IMPORTANCE PIE CHARTS
# ===============================================================================

def _add_pie_leaders(ax, wedges, labels, pcts, threshold=4.0,
                     inner_r=0.72, outer_r=1.18, text_r=1.32):
    """Label large slices inside (white); draw leader lines for small slices."""
    for wedge, label, pct in zip(wedges, labels, pcts):
        if pct < 0.3:
            continue
        ang    = (wedge.theta1 + wedge.theta2) / 2.0
        rad    = math.radians(ang)
        ca, sa = math.cos(rad), math.sin(rad)
        if pct >= threshold:
            ax.text(inner_r * ca, inner_r * sa, f'{pct:.1f}%',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
        else:
            ha = 'left' if ca >= 0 else 'right'
            ax.annotate(
                f'{pct:.1f}%',
                xy=(0.97 * ca, 0.97 * sa),
                xytext=((text_r + 0.08) * ca, text_r * sa),
                fontsize=10, color='#222', ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color='#666', lw=0.85,
                                connectionstyle='arc3,rad=0.0'),
            )


def _draw_meta_brackets(ax, wedges, group_names, arc_r=1.55, text_r=1.78):
    """
    Draw curved bracket arcs outside the left pie chart to indicate
    Structure-Based vs Sequence-Based meta-groupings.

    Each meta-group gets a coloured arc with radial tick marks at the
    endpoints and a centred text label — suitable for academic figures.
    """
    from matplotlib.patches import Arc

    # Build contiguous angular spans for each meta-group
    meta_spans = {}   # meta_name -> (min_theta1, max_theta2)
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

        # Draw the arc
        arc = Arc((0, 0), 2 * arc_r, 2 * arc_r,
                  angle=0, theta1=ang_start, theta2=ang_end,
                  color=clr, lw=2.2, linestyle='-')
        ax.add_patch(arc)

        # Radial tick marks at both endpoints
        for ang_deg in [ang_start, ang_end]:
            rad = math.radians(ang_deg)
            ca, sa = math.cos(rad), math.sin(rad)
            ax.plot(
                [(arc_r - tick_len) * ca, (arc_r + tick_len) * ca],
                [(arc_r - tick_len) * sa, (arc_r + tick_len) * sa],
                color=clr, lw=2.2, solid_capstyle='round',
            )

        # Label at the midpoint of the arc
        mid_deg = (ang_start + ang_end) / 2.0
        mid_rad = math.radians(mid_deg)
        tx = text_r * math.cos(mid_rad)
        ty = text_r * math.sin(mid_rad)
        ha = 'left' if math.cos(mid_rad) >= 0 else 'right'
        ax.text(tx, ty, meta_name, ha=ha, va='center',
                fontsize=12, fontweight='bold', color=clr,
                fontstyle='italic')


def plot_feature_importance_pies(imp_df, out_dir, model_type, pdb_id=None):
    """
    Produce TWO separate figures so each pie chart is full-size:

      Figure 1  –  Feature Categories pie  (sub-categories + meta-group arcs)
      Figure 2  –  Individual Features pie (top-20, legend text coloured by
                    Structure-Based vs Sequence-Based)

    Negative importances are clipped to zero and re-normalised to 100%.
    """
    if imp_df is None or imp_df.empty:
        return

    imp_df = imp_df.copy()

    # ---- Clip negatives & re-normalise to 100% -----------------------------
    imp_df['Imp (%)'] = imp_df['Imp (%)'].clip(lower=0)
    total = imp_df['Imp (%)'].sum()
    if total > 0:
        imp_df['Imp (%)'] = imp_df['Imp (%)'] / total * 100.0

    imp_df['Group'] = imp_df['Feature'].apply(_classify_feature)

    # ---- category pie data --------------------------------------------------
    group_imp = (imp_df.groupby('Group')['Imp (%)']
                 .sum().reindex(GROUP_ORDER).fillna(0))
    group_imp = group_imp[group_imp > 0]
    palette1  = [GROUP_PALETTE[GROUP_ORDER.index(g)] for g in group_imp.index]
    g_labels  = [g.replace('\n', ' ') for g in group_imp.index]

    meta_totals = {}
    for grp, val in group_imp.items():
        meta = META_GROUPS.get(grp, 'Other')
        meta_totals[meta] = meta_totals.get(meta, 0.0) + val

    # ---- individual feature pie data ----------------------------------------
    top_n     = 20
    imp_sort  = imp_df.sort_values('Imp (%)', ascending=False)
    top_df    = imp_sort.head(top_n)
    other_sum = imp_sort.iloc[top_n:]['Imp (%)'].sum()
    labels2   = list(top_df['Feature'])
    sizes2    = list(top_df['Imp (%)'])
    groups2   = list(top_df['Group'])       # keep group for text colouring
    if other_sum > 0.01:
        labels2.append(f'Others ({len(imp_sort) - top_n} feats)')
        sizes2.append(other_sum)
        groups2.append(None)
    cmap2   = plt.colormaps['tab20b'].resampled(len(labels2))
    colors2 = [cmap2(i) for i in range(len(labels2))]

    title_suffix = (f'  |  Structure: {pdb_id}' if pdb_id else '')
    prefix       = f'{pdb_id}_' if pdb_id else ''

    # =====================================================================
    # FIGURE 1 — Feature Categories
    # =====================================================================
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
        fontsize=20, fontweight='bold')

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

    # Meta-group bracket arcs
    _draw_meta_brackets(ax_pie1, wedges1, list(group_imp.index))

    # Legend below: sub-categories, then summary at bottom (no separator)
    phys_val = meta_totals.get('Structure-Based', 0.0)
    seq_val  = meta_totals.get('Sequence-Based', 0.0)

    leg_handles = [plt.Rectangle((0, 0), 1, 1, fc=palette1[i], ec='white', lw=1.2)
                   for i in range(len(g_labels))]
    leg_labels  = [f'{lbl}  ({pct:.1f}%)'
                   for lbl, pct in zip(g_labels, group_imp.values)]

    # Summary entries at the end (same META colours as arcs — blue/purple)
    leg_handles.append(plt.Rectangle((0, 0), 1, 1,
                       fc=META_PALETTE['Structure-Based'], ec='white', lw=1.2))
    leg_labels.append(f'Structure-Based: {phys_val:.1f}%')
    leg_handles.append(plt.Rectangle((0, 0), 1, 1,
                       fc=META_PALETTE['Sequence-Based'], ec='white', lw=1.2))
    leg_labels.append(f'Sequence-Based: {seq_val:.1f}%')

    ax_leg1.legend(
        leg_handles, leg_labels,
        loc='upper center', ncol=3,
        fontsize=12, frameon=True, framealpha=0.93,
        title='Feature Sub-Category', title_fontsize=14,
        handlelength=1.8, handleheight=1.5, labelspacing=0.7,
        columnspacing=2.0,
    )

    fname1 = f'{prefix}{model_type.lower()}_importance_categories.png'
    fig1.savefig(os.path.join(out_dir, fname1), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'  [PIE] Category importance  -> {fname1}')

    # =====================================================================
    # FIGURE 2 — Individual Features (Top 20)
    #   Legend text is coloured by Structure-Based (blue) vs Sequence-Based (red).
    # =====================================================================
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
        fontsize=20, fontweight='bold')

    wedges2, _ = ax_pie2.pie(
        sizes2, colors=colors2, startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
        labels=None, autopct=None,
    )
    _add_pie_leaders(ax_pie2, wedges2, labels2, sizes2,
                     threshold=3.5, inner_r=0.65, outer_r=1.12, text_r=1.28)
    ax_pie2.set_xlim(-1.55, 1.55)
    ax_pie2.set_ylim(-1.55, 1.55)

    # Build legend with text coloured by meta-group
    leg = ax_leg2.legend(
        wedges2, labels2,
        loc='upper center', ncol=4,
        fontsize=12, frameon=True, framealpha=0.93,
        title=f'Top {top_n} Features', title_fontsize=14,
        handlelength=1.3, handleheight=1.3, labelspacing=0.6,
        columnspacing=1.5,
    )
    # Colour each legend text by its meta-group
    for txt_obj, grp in zip(leg.get_texts(), groups2):
        meta = META_GROUPS.get(grp, 'Other') if grp else 'Other'
        clr  = META_PALETTE.get(meta, '#333')
        txt_obj.set_color(clr)
        txt_obj.set_fontweight('bold')

    # Colour key below the main legend
    phys_clr = META_PALETTE['Structure-Based']
    seq_clr  = META_PALETTE['Sequence-Based']
    key_text = (f'Legend text colour:   '
                f'Structure-Based   |   '
                f'Sequence-Based')
    ax_leg2.text(
        0.5, 0.02, key_text,
        transform=ax_leg2.transAxes, ha='center', va='bottom',
        fontsize=13, fontweight='bold',
    )
    # Manually colour each part via overlaid coloured fragments
    ax_leg2.annotate(
        'Structure-Based', xy=(0.44, 0.02), xycoords='axes fraction',
        fontsize=13, fontweight='bold', color=phys_clr, ha='center', va='bottom',
    )
    ax_leg2.annotate(
        '|', xy=(0.575, 0.02), xycoords='axes fraction',
        fontsize=13, color='#666', ha='center', va='bottom',
    )
    ax_leg2.annotate(
        'Sequence-Based', xy=(0.66, 0.02), xycoords='axes fraction',
        fontsize=13, fontweight='bold', color=seq_clr, ha='center', va='bottom',
    )
    # Blank out the mono-colour version and replace with label only
    ax_leg2.texts[0].set_alpha(0)   # hide the plain key_text

    # Small "Legend text colour:" label in grey
    ax_leg2.annotate(
        'Legend text colour:', xy=(0.28, 0.02), xycoords='axes fraction',
        fontsize=12, color='#555', ha='center', va='bottom', fontstyle='italic',
    )

    fname2 = f'{prefix}{model_type.lower()}_importance_individual.png'
    fig2.savefig(os.path.join(out_dir, fname2), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'  [PIE] Individual importance -> {fname2}')


# ===============================================================================
# TRAINING
# ===============================================================================

def run_training(model_type, args, X, y, groups, full_df):
    # Verify model is available before doing anything
    test_model = get_model(model_type, args.lr, args.depth)
    if test_model is None:
        print(f"[SKIP] Model '{model_type}' is not available. "
              f"(lightgbm not installed?)")
        return {"Model": model_type.upper(), "MAE_Shift": float('inf')}

    output_dir      = os.path.join("MCCE_ML-models", f"model_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    log_file_path   = os.path.join(output_dir, "training_log.txt")
    original_stdout = sys.stdout
    sys.stdout      = Logger(log_file_path)

    # 1. 80/20 Grouped Split
    rng         = check_random_state(args.seed)
    unique_pdbs = groups.unique()
    rng.shuffle(unique_pdbs)
    split_idx   = int(len(unique_pdbs) * 0.8)
    train_pdbs, val_pdbs = unique_pdbs[:split_idx], unique_pdbs[split_idx:]
    X_dev  = X[groups.isin(train_pdbs)];  y_dev  = y[groups.isin(train_pdbs)]
    g_dev  = groups[groups.isin(train_pdbs)]
    X_val  = X[groups.isin(val_pdbs)];    y_val  = y[groups.isin(val_pdbs)]
    df_dev = full_df[groups.isin(train_pdbs)]
    df_val = full_df[groups.isin(val_pdbs)]
    print(f"TRAIN COMPOSITION: {get_dist_string(df_dev)}")
    print(f"VAL COMPOSITION:   {get_dist_string(df_val)}")

    # 2. 5-Fold CV
    oof_preds    = np.zeros(len(y_dev))
    all_cv_evals = []
    gkf = GroupKFold(n_splits=5)
    print("\n" + "-"*110 + "\nINDIVIDUAL K-FOLD STATISTICS (DEVELOPMENT POOL)\n" + "-"*110)
    print(f"{'Fold':<5} | {'R2':<7} | {'MAE':<7} | {'RMSE':<7} | "
          f"{'+-1.0':<7} | {'+-2.0':<7} | {'+-3.0':<7}")

    for fold, (tr, te) in enumerate(gkf.split(X_dev, y_dev, groups=g_dev), 1):
        model       = get_model(model_type, args.lr, args.depth)
        X_tr, X_te  = X_dev.iloc[tr], X_dev.iloc[te]
        y_tr, y_te  = y_dev.iloc[tr], y_dev.iloc[te]
        if model_type == 'lgbm':
            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)
            model.named_steps['lgbm'].fit(
                X_tr_s, y_tr,
                eval_set=[(X_tr_s, y_tr), (X_te_s, y_te)],
                eval_names=['training', 'valid_0'],
                callbacks=[early_stopping(stopping_rounds=50),
                           log_evaluation(period=0)])
            all_cv_evals.append(model.named_steps['lgbm'].evals_result_)
            preds = model.named_steps['lgbm'].predict(X_te_s)
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            if model_type == 'mlp':
                all_cv_evals.append(model.named_steps['mlp'].loss_curve_)
        s = get_full_stats(y_te, preds)
        oof_preds[te] = preds
        print(f"{fold:<5} | {s['R2']:<7.3f} | {s['MAE']:<7.3f} | "
              f"{s['RMSE']:<7.3f} | {s['pm1']:<7.1f} | "
              f"{s['pm2']:<7.1f} | {s['pm3']:<7.1f}")

    if all_cv_evals:
        plot_combined_cv_loss(all_cv_evals, model_type, output_dir)

    # 3. Final Model
    print(f"\nFinalizing production model on the full 80% training pool...")
    final_model = get_model(model_type, args.lr, args.depth)
    final_model.fit(X_dev, y_dev)
    val_p_shift = final_model.predict(X_val)
    dev_p_shift = final_model.predict(X_dev)

    create_dashboard(y_dev.values, dev_p_shift, y_val.values, val_p_shift,
                     df_dev, df_val, model_type, output_dir, mode="shift")
    if 'pKa0' in df_dev.columns:
        dev_y_pka = df_dev['pKa0'].values + y_dev.values
        dev_p_pka = df_dev['pKa0'].values + dev_p_shift
        val_y_pka = df_val['pKa0'].values + y_val.values
        val_p_pka = df_val['pKa0'].values + val_p_shift
        create_dashboard(dev_y_pka, dev_p_pka, val_y_pka, val_p_pka,
                         df_dev, df_val, model_type, output_dir, mode="pka")

    # 4. Feature Importance
    print(f"\nRANKED PHYSICS DRIVERS ({model_type.upper()}):\n" + "-"*60)
    m_i = (final_model.named_steps[model_type]
           if hasattr(final_model, 'named_steps') else final_model)
    imp_vals, method_info = None, ""

    if hasattr(m_i, 'feature_importances_'):
        imp_vals    = m_i.feature_importances_
        method_info = "Feature Importance (Gini/Gain)"
    elif hasattr(m_i, 'coef_'):
        imp_vals    = np.abs(m_i.coef_)
        method_info = "Absolute Coefficients (Sensitivity)"
    else:
        try:
            print(f"[PROCESS] Calculating Permutation Importance "
                  f"for {model_type.upper()}...")
            r = permutation_importance(final_model, X_val, y_val,
                                       n_repeats=5, random_state=42)
            imp_vals    = r.importances_mean
            method_info = "Permutation Importance (Validation Holdout)"
        except Exception as e:
            print(f"[UNAVAILABLE] Cannot calculate for {model_type.upper()}: {e}")

    if imp_vals is not None:
        if np.sum(imp_vals) == 0:
            print(f"[NOTICE] All importance values zero for {model_type.upper()}.")
        else:
            imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Imp (%)': np.round((imp_vals / np.sum(imp_vals)) * 100, 2),
            }).sort_values('Imp (%)', ascending=False)
            print(f"Ranking Method: {method_info}")
            print(imp_df.head(15).to_string(index=False))
            imp_df.to_csv(
                os.path.join(output_dir, 'feature_importance_full.csv'),
                index=False)
            plot_feature_importance_pies(imp_df, output_dir, model_type)

    # 5. Save artifacts
    print(f"\n{'='*85}\nARTIFACTS MANIFEST: {model_type.upper()}\n{'='*85}")
    manifest = [
        "pka_model.pkl", "model_features.pkl",
        "performance_correlation.png", "performance_correlation_pKa.png",
        "loss_curve_cv_combined.png", "training_log.txt",
        "feature_importance_full.csv",
        f"{model_type.lower()}_importance_categories.png",
        f"{model_type.lower()}_importance_individual.png",
        "holdout_analysis.csv",
    ]
    for i, f in enumerate(manifest, 1):
        print(f"{i}. {f:<42}: "
              f"{os.path.abspath(os.path.join(output_dir, f))}")

    joblib.dump(final_model, os.path.join(output_dir, 'pka_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(output_dir, 'model_features.pkl'))
    h_df = df_val.copy()
    h_df['Predicted_pKa_shift'] = val_p_shift
    h_df.to_csv(os.path.join(output_dir, 'holdout_analysis.csv'), index=False)

    sys.stdout = original_stdout

    # 6. External test (optional)
    if os.path.exists(args.test_data):
        run_test_evaluation(final_model, X.columns.tolist(),
                            model_type, output_dir, args.test_data)
    else:
        print(f"  [INFO] Test data '{args.test_data}' not found -- "
              f"skipping external evaluation.")

    return {
        "Model":     model_type.upper(),
        "MAE_Shift": mean_absolute_error(y_val, val_p_shift),
    }


# ===============================================================================
# COMPARISON  --  DATA LOADERS
# ===============================================================================

def find_model_results(base_dir):
    """Return {MODEL_NAME: holdout_csv_path} for all trained models."""
    results = {}
    if not os.path.isdir(base_dir):
        return results
    for folder in os.listdir(base_dir):
        if not folder.startswith("model_"):
            continue
        model_name = folder.upper().replace("MODEL_", "")
        csv_path   = os.path.join(base_dir, folder, "holdout_analysis.csv")
        if os.path.exists(csv_path):
            results[model_name] = csv_path
    return results


def find_test_results(base_dir):
    """Return {test_name: {MODEL_NAME: test_predictions_csv}}."""
    test_sets = {}
    if not os.path.isdir(base_dir):
        return test_sets
    for folder in os.listdir(base_dir):
        if not folder.startswith("model_"):
            continue
        model_name = folder.upper().replace("MODEL_", "")
        model_dir  = os.path.join(base_dir, folder)
        for item in os.listdir(model_dir):
            if (item.startswith("test_") and
                    os.path.isdir(os.path.join(model_dir, item))):
                test_name = item[5:]
                pred_csv  = os.path.join(model_dir, item, "test_predictions.csv")
                if os.path.exists(pred_csv):
                    test_sets.setdefault(test_name, {})[model_name] = pred_csv
    return test_sets


def find_importance_csvs(base_dir):
    """Return {MODEL_NAME: feature_importance_full.csv} for all trained models."""
    results = {}
    if not os.path.isdir(base_dir):
        return results
    for folder in os.listdir(base_dir):
        if not folder.startswith("model_"):
            continue
        model_name = folder.upper().replace("MODEL_", "")
        csv_path   = os.path.join(base_dir, folder, "feature_importance_full.csv")
        if os.path.exists(csv_path):
            results[model_name] = csv_path
    return results


def load_holdout_data(model_files):
    all_data = []
    for model_name in MODEL_ORDER:
        if model_name not in model_files:
            continue
        try:
            df = pd.read_csv(model_files[model_name])
            if 'resi_type' not in df.columns:
                df['resi_type'] = df['resi+chainId+resid'].str[:3]
            if 'Target_pKa' not in df.columns:
                df['Target_pKa']    = df['pKa0'] + df['Target_pKa_shift']
                df['Predicted_pKa'] = df['pKa0'] + df['Predicted_pKa_shift']
            df['Residual_Shift'] = df['Predicted_pKa_shift'] - df['Target_pKa_shift']
            df['Model'] = model_name
            all_data.append(df)
            print(f"  Loaded {model_name:<10}: {len(df):>5} residues")
        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
    return all_data


def load_test_data(test_model_files):
    all_data = []
    for model_name in MODEL_ORDER:
        if model_name not in test_model_files:
            continue
        try:
            df = pd.read_csv(test_model_files[model_name])
            if 'resi_type' not in df.columns:
                df['resi_type'] = df['resi+chainId+resid'].str[:3]
            if ('Predicted_pKa_shift' not in df.columns and
                    'Predicted_pKa' in df.columns):
                df['Predicted_pKa_shift'] = df['Predicted_pKa'] - df['pKa0']
            if 'Target_pKa' not in df.columns and 'pKa0' in df.columns:
                df['Target_pKa']    = df['pKa0'] + df['Target_pKa_shift']
                df['Predicted_pKa'] = df['pKa0'] + df['Predicted_pKa_shift']
            df['Residual_Shift'] = df['Predicted_pKa_shift'] - df['Target_pKa_shift']
            df['Model'] = model_name
            all_data.append(df)
            print(f"  Loaded {model_name:<10}: {len(df):>5} residues")
        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
    return all_data


# ===============================================================================
# COMPARISON  --  PLOTS
# ===============================================================================

def plot_comparison_correlation(all_data, mode, plot_dir,
                                label_prefix="", fname_prefix="compare"):
    lims  = (-10, 10) if mode == 'shift' else (0, 14)
    x_col = 'Target_pKa_shift'    if mode == 'shift' else 'Target_pKa'
    y_col = 'Predicted_pKa_shift' if mode == 'shift' else 'Predicted_pKa'
    label = "pKa Shift" if mode == 'shift' else "Absolute pKa"

    stats_rows = []
    for m_df in all_data:
        m_name = m_df['Model'].iloc[0]
        if x_col not in m_df.columns or y_col not in m_df.columns:
            continue
        stats_rows.append((m_name, get_full_stats(m_df[x_col], m_df[y_col])))

    stats_text = (
        f"COMPARATIVE {label_prefix}{label.upper()} PERFORMANCE DASHBOARD\n"
        f"{'='*84}\n"
        f"  {'Model':<10} | {'N':>5} | {'MAE':>7} | {'RMSE':>7} | {'R2':>7} | "
        f"{'+-1.0 pH':>8} | {'+-2.0 pH':>8} | {'+-3.0 pH':>8}\n"
        f"  {'-'*82}\n"
    )
    for m_name, s in stats_rows:
        stats_text += (
            f"  {m_name:<10} | {s['N']:>5} | {s['MAE']:>7.3f} | "
            f"{s['RMSE']:>7.3f} | {s['R2']:>7.3f} | "
            f"{s['pm1']:>7.1f}% | {s['pm2']:>7.1f}% | {s['pm3']:>7.1f}%\n"
        )
    stats_text += (
        f"  {'-'*82}\n"
        f"  Residue dist: {get_dist_string(all_data[0])}\n"
        f"{'='*84}"
    )

    n_models = len(stats_rows)
    text_h   = max(1.3, (n_models + 5) * 0.23)
    fig      = plt.figure(figsize=(16, 18 + text_h))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[5, text_h], hspace=0.18)
    ax       = fig.add_subplot(gs_outer[0])
    ax_text  = fig.add_subplot(gs_outer[1])
    ax_text.axis("off")

    sns.set_style("whitegrid", {"axes.edgecolor": "black", "grid.color": ".85"})
    x_range = np.linspace(lims[0], lims[1], 100)
    ax.plot(lims, lims, color='black', lw=3, label='Identity', zorder=1)
    for i, (ls, alpha) in enumerate([('--', 0.4), (':', 0.22)], 1):
        ax.plot(x_range, x_range + i, color='gray', linestyle=ls,
                alpha=alpha, zorder=1)
        ax.plot(x_range, x_range - i, color='gray', linestyle=ls,
                alpha=alpha, zorder=1)

    for m_df in all_data:
        m_name = m_df['Model'].iloc[0]
        if x_col not in m_df.columns or y_col not in m_df.columns:
            continue
        s = get_full_stats(m_df[x_col], m_df[y_col])
        ax.scatter(m_df[x_col], m_df[y_col],
                   color=COLORS.get(m_name, '#333'), edgecolor='white',
                   linewidth=0.8, s=85, alpha=0.85,
                   label=f"{m_name}  MAE={s['MAE']:.3f}  R2={s['R2']:.3f}",
                   zorder=5)
        sns.regplot(x=x_col, y=y_col, data=m_df, scatter=False, ax=ax,
                    line_kws={'lw': 3.2, 'alpha': 0.9, 'zorder': 4},
                    color=COLORS.get(m_name, '#333'), ci=None)

    ax.set_title(f"{label_prefix}Multi-Model {label} Comparison",
                 fontsize=24, fontweight='bold')
    ax.set_xlabel(f"MCCE {label}", fontsize=18)
    ax.set_ylabel(f"ML Predicted {label}", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(loc='lower right', frameon=True, fontsize=13)
    txt = ax_text.text(0.5, 0.97, stats_text, transform=ax_text.transAxes,
                       family='monospace', fontsize=11,
                       verticalalignment='top', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='#f9f9f9',
                                 alpha=0.97, edgecolor='black'))

    # Stretch the textbox to match the full plot width after drawing
    fig.canvas.draw()
    renderer  = fig.canvas.get_renderer()
    ax_bbox   = ax.get_window_extent(renderer=renderer)       # plot axes width
    txt_bbox  = txt.get_window_extent(renderer=renderer)      # current text bbox
    # Scale font so text fills the axes width (capped to avoid over-expansion)
    current_w = txt_bbox.width
    target_w  = ax_bbox.width * 0.98
    if current_w > 0:
        scale = min(target_w / current_w, 1.6)               # don't go huge
        txt.set_fontsize(11 * scale)

    out_path = os.path.join(plot_dir, f"{fname_prefix}_correlation_{mode}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] {os.path.basename(out_path)}")


def plot_comparison_residue_mae(all_data, plot_dir, fname_prefix="compare"):
    rows = []
    for m_df in all_data:
        m_name = m_df['Model'].iloc[0]
        for res, group in m_df.groupby('resi_type'):
            rows.append({
                'Model':   m_name,
                'Residue': res,
                'MAE':     mean_absolute_error(group['Target_pKa_shift'],
                                               group['Predicted_pKa_shift']),
            })
    if not rows:
        return
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Residue', y='MAE', hue='Model', data=pd.DataFrame(rows),
                palette=COLORS, edgecolor='black', linewidth=1.4)
    plt.title('MAE for pKa Shift by Residue Type', fontsize=20, fontweight='bold')
    plt.ylabel('MAE (pH units)', fontsize=16)
    plt.xlabel('Residue Type', fontsize=16)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='+-1.0 pH')
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(fontsize=12); plt.grid(axis='y', alpha=0.35); plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{fname_prefix}_residue_mae.png")
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"  [PLOT] {os.path.basename(out_path)}")


def run_significance_tests(all_data, label=""):
    if len(all_data) < 2:
        return
    print(f"\n{'='*95}\nPAIRED SIGNIFICANCE TESTING {label}\n{'='*95}")
    print(f"{'Model A':<12} vs {'Model B':<12} | {'p-value':>10} | "
          f"Significant (a=0.05)")
    print("-"*60)
    for i in range(len(all_data)):
        for j in range(i + 1, len(all_data)):
            m1, m2 = all_data[i], all_data[j]
            common = pd.merge(
                m1[['resi+chainId+resid', 'Residual_Shift']],
                m2[['resi+chainId+resid', 'Residual_Shift']],
                on='resi+chainId+resid', suffixes=('_1', '_2'),
            )
            if len(common) < 3:
                continue
            _, p_val = stats.ttest_rel(
                common['Residual_Shift_1'].abs(),
                common['Residual_Shift_2'].abs(),
            )
            sig = "YES" if p_val < 0.05 else "NO"
            print(f"{m1['Model'].iloc[0]:<12} vs {m2['Model'].iloc[0]:<12} | "
                  f"{p_val:>10.4f} | {sig}")


def _load_importance(imp_csv):
    try:
        df = pd.read_csv(imp_csv)
        df['Group'] = df['Feature'].apply(_classify_feature)
        return df
    except Exception:
        return None


def plot_cross_model_group_bar(importance_csvs, plot_dir):
    """Stacked horizontal bar: feature-group importance across all models."""
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in importance_csvs:
            continue
        imp_df = _load_importance(importance_csvs[model_name])
        if imp_df is None or imp_df.empty:
            continue
        imp_df = imp_df.drop(columns=['Group'], errors='ignore')
        # Clip negatives & re-normalise (permutation importance can be < 0)
        imp_df['Imp (%)'] = imp_df['Imp (%)'].clip(lower=0)
        total = imp_df['Imp (%)'].sum()
        if total > 0:
            imp_df['Imp (%)'] = imp_df['Imp (%)'] / total * 100.0
        imp_df['Group'] = imp_df['Feature'].apply(_classify_feature)
        group_imp = (imp_df.groupby('Group')['Imp (%)']
                     .sum().reindex(GROUP_ORDER).fillna(0))
        row = {'Model': model_name}
        row.update(group_imp.to_dict())
        rows.append(row)
    if not rows:
        return
    bar_df = pd.DataFrame(rows).set_index('Model')
    cols   = [g for g in GROUP_ORDER if g in bar_df.columns]
    bar_df = bar_df[cols]
    colors = [GROUP_PALETTE[GROUP_ORDER.index(g)] for g in cols]
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.9 + 1.5)))
    bar_df.plot(kind='barh', stacked=True, ax=ax,
                color=colors, edgecolor='white', linewidth=0.6)
    ax.set_xlabel('Feature Group Importance (%)', fontsize=16)
    ax.set_title('Cross-Model Feature Group Importance',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend([g.replace('\n', ' ') for g in cols],
              loc='lower right', fontsize=11, frameon=True,
              title='Feature Group', title_fontsize=12)
    ax.set_xlim(0, 100); plt.tight_layout()
    out_path = os.path.join(plot_dir, 'cross_model_feature_groups.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f'  [PLOT] cross_model_feature_groups.png')


def plot_per_model_importance_pies(importance_csvs, plot_dir):
    """Per-model two-panel pie chart in plot_dir/feature_importance/."""
    imp_dir = os.path.join(plot_dir, "feature_importance")
    os.makedirs(imp_dir, exist_ok=True)
    for model_name in MODEL_ORDER:
        if model_name not in importance_csvs:
            continue
        imp_df = _load_importance(importance_csvs[model_name])
        if imp_df is None or imp_df.empty:
            continue
        imp_df = imp_df.drop(columns=['Group'], errors='ignore')
        plot_feature_importance_pies(imp_df, imp_dir, model_name)


def print_leaderboard(holdout_data, test_sets, base_dir, plot_dir):
    rows = []
    for m_df in holdout_data:
        m_name = m_df['Model'].iloc[0]
        s = get_full_stats(m_df['Target_pKa_shift'], m_df['Predicted_pKa_shift'])
        rows.append({'Model': m_name, 'Dataset': 'Holdout', **s})
    for test_name, test_model_files in find_test_results(base_dir).items():
        for m_df in load_test_data(test_model_files):
            m_name = m_df['Model'].iloc[0]
            s = get_full_stats(m_df['Target_pKa_shift'],
                               m_df['Predicted_pKa_shift'])
            rows.append({'Model': m_name, 'Dataset': test_name, **s})
    if not rows:
        return
    lb = (pd.DataFrame(rows)
          .sort_values(['Dataset', 'MAE'])
          .rename(columns={'pm1': '+-1.0%', 'pm2': '+-2.0%', 'pm3': '+-3.0%'}))
    print(f"\n{'='*90}\nGLOBAL LEADERBOARD\n{'='*90}")
    print(lb[['Model', 'Dataset', 'N', 'MAE', 'RMSE', 'R2',
              '+-1.0%', '+-2.0%']].to_string(index=False))
    csv_out = os.path.join(plot_dir, "global_leaderboard.csv")
    lb.to_csv(csv_out, index=False)
    print(f"\n  Leaderboard saved -> {csv_out}")


# ===============================================================================
# COMPARISON  --  ORCHESTRATOR
# ===============================================================================

def run_comparison(base_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_style("whitegrid", {"axes.edgecolor": "black", "grid.color": ".8"})

    # ---- Holdout ------------------------------------------------------------
    model_files  = find_model_results(base_dir)
    holdout_data = []

    if model_files:
        print(f"\n{'#'*85}\nCOMPARISON: HOLDOUT PERFORMANCE\n{'#'*85}")
        holdout_data = load_holdout_data(model_files)
        if holdout_data:
            for mode in ['shift', 'absolute']:
                plot_comparison_correlation(holdout_data, mode, plot_dir,
                                            label_prefix="Holdout: ",
                                            fname_prefix="compare")
            plot_comparison_residue_mae(holdout_data, plot_dir,
                                        fname_prefix="compare")
            run_significance_tests(holdout_data, label="HOLDOUT")
    else:
        print(f"[WARN] No trained model results found in '{base_dir}'. "
              f"Run mcce2ml_2-train.py -m <model> first.")
        return

    # ---- External test sets -------------------------------------------------
    test_sets = find_test_results(base_dir)
    for test_name, test_model_files in test_sets.items():
        print(f"\n{'#'*85}\n"
              f"COMPARISON: EXTERNAL TEST -- {test_name}\n{'#'*85}")
        test_data = load_test_data(test_model_files)
        if not test_data:
            continue
        test_plot_dir = os.path.join(plot_dir, f"test_{test_name}")
        os.makedirs(test_plot_dir, exist_ok=True)
        for mode in ['shift', 'absolute']:
            plot_comparison_correlation(test_data, mode, test_plot_dir,
                                        label_prefix=f"Test ({test_name}): ",
                                        fname_prefix=f"test_{test_name}")
        plot_comparison_residue_mae(test_data, test_plot_dir,
                                    fname_prefix=f"test_{test_name}")
        run_significance_tests(test_data, label=f"TEST ({test_name})")

    # ---- Feature importance pies --------------------------------------------
    importance_csvs = find_importance_csvs(base_dir)
    if importance_csvs:
        print(f"\n{'#'*85}\nFEATURE IMPORTANCE VISUALISATIONS\n{'#'*85}")
        plot_per_model_importance_pies(importance_csvs, plot_dir)
        plot_cross_model_group_bar(importance_csvs, plot_dir)
    else:
        print(f"\n[INFO] No feature_importance_full.csv found -- "
              f"run mcce2ml_2-train.py to generate importance data.")

    # ---- Global leaderboard -------------------------------------------------
    print_leaderboard(holdout_data, test_sets, base_dir, plot_dir)

    print(f"\n{'='*85}\nCOMPARISON COMPLETE\n"
          f"All plots saved to '{plot_dir}/'\n{'='*85}")


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "mcce2ml_2-train: Train MCCE-ML models and compare results.\n\n"
            "Training mode (default):\n"
            "  python mcce2ml_2-train.py -m lgbm\n"
            "  python mcce2ml_2-train.py --all\n"
            "  python mcce2ml_2-train.py --all --skip_compare\n\n"
            "Comparison-only mode (no training):\n"
            "  python mcce2ml_2-train.py --compare\n"
            "  python mcce2ml_2-train.py --compare "
            "--base_dir MCCE_ML-models --out_dir comparison_plots\n"
        ),
    )

    # ---- Comparison-only flag -----------------------------------------------
    parser.add_argument(
        "--compare", action="store_true",
        help="Run cross-model comparison only (no training).",
    )

    # ---- Training arguments -------------------------------------------------
    parser.add_argument("-i", "--input", default="ml_dataset.pkl",
                        help="Training data pickle (default: ml_dataset.pkl)")
    parser.add_argument("-m", "--model", default="lgbm",
                        help="Architecture: lgbm, rf, svr, bayesian, mlp, elastic, knn")
    parser.add_argument("--all", action="store_true",
                        help="Train ALL architectures sequentially")
    parser.add_argument("-r", "--seed", type=int, default=42,
                        help="Random seed for protein split (default: 42)")
    parser.add_argument("-l", "--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("-d", "--depth", type=int, default=5,
                        help="Max tree depth (default: 5)")
    parser.add_argument("-t", "--test_data", type=str,
                        default="test_ml_dataset.pkl",
                        help="External test set pickle "
                             "(default: test_ml_dataset.pkl)")
    parser.add_argument("--skip_compare", action="store_true",
                        help="Skip cross-model comparison after training")

    # ---- Comparison arguments (shared with --compare mode) ------------------
    parser.add_argument("--base_dir", default="MCCE_ML-models",
                        help="Directory with model_* subdirs "
                             "(default: MCCE_ML-models)")
    parser.add_argument("--out_dir", default="comparison_plots",
                        help="Output directory for comparison plots "
                             "(default: comparison_plots)")

    args = parser.parse_args()

    # ---- Comparison-only mode -----------------------------------------------
    if args.compare:
        print(f"\n{'='*85}")
        print(f"  MCCE-ML Cross-Model Comparison")
        print(f"  Base dir  : {os.path.abspath(args.base_dir)}")
        print(f"  Output dir: {os.path.abspath(args.out_dir)}")
        print(f"{'='*85}")
        run_comparison(args.base_dir, args.out_dir)
        return

    # ---- Training mode ------------------------------------------------------
    if not os.path.exists(args.input):
        print(f"ERROR: Training data '{args.input}' not found. "
              f"Run mcce2ml_1-features.py first.")
        return

    df     = pd.read_pickle(args.input)
    df_ml  = pd.get_dummies(df, columns=['resi_type'])
    X      = df_ml.drop(columns=['PDB', 'resi+chainId+resid', 'Target_pKa_shift'])
    y      = df_ml['Target_pKa_shift']
    groups = df_ml['PDB']

    to_run  = (['lgbm', 'rf', 'svr', 'bayesian', 'mlp', 'elastic', 'knn']
               if args.all else [args.model])
    results = [run_training(m, args, X, y, groups, df_ml) for m in to_run]

    if args.all:
        print("\n" + "#"*30 + "\nTRAINING LEADERBOARD (VAL SET)\n" + "#"*30)
        print(pd.DataFrame(results).sort_values("MAE_Shift").to_string(index=False))

    if not args.skip_compare:
        print(f"\n{'#'*85}\nSTARTING CROSS-MODEL COMPARISON\n{'#'*85}")
        try:
            run_comparison(args.base_dir, args.out_dir)
        except Exception as e:
            print(f"\n[ERROR] Comparison failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
