#!/usr/bin/env python3
"""
Publication-style plots of pH-dependent relative free energies (kcal/mol)
and analogous Henderson-Hasselbalch fractional occupancies.

Each curve is labelled directly on the graph with its species
(A-, HA, BH+ or B), and the governing equations are collected in the
figure legend.

All outputs are written into the ``plot_acid_base_with_equations/``
directory:
    figure_ionized_states_equations.png / .pdf
    figure_neutral_states_equations.png / .pdf
    acid_base_values.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================= User settings ============================= #

PKA = 7.0
TEMPERATURE_K = 298.15
PH_MIN = 0.0
PH_MAX = 14.0
N_POINTS = 1401
DPI = 300

# When True, the equation legend is omitted from every figure (--no_legend).
NO_LEGEND = False

# Font sizes for the in-panel text, collected here so the whole figure family
# can be scaled from one place. Axis labels and tick labels are set through
# rcParams below; the rest are passed explicitly where the text is drawn.
FS_PANEL_LABEL = 26     # the bold A/B/C/D identifier
FS_REACTION = 16        # the reaction equilibrium scheme
FS_EQUATION = 13        # the dG / f equations (hidden by --no_legend)
FS_CURVE_LABEL = 14     # the on-curve species labels
FS_PKA = 13             # the "pKa = 7" marker annotation
FS_REFERENCE = 13       # the "HA_{ΔG=0}" label on the zero line
ZERO_LINE_WIDTH = 2.0    # the bold black ΔG = 0 line (diagonal is 2.6)

# Translucent white backing, so a text block stays legible where the larger
# type runs up against a curve (the equations graze the sigmoid flank on the
# ascending panels). Same treatment as the reaction scheme.
EQ_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2)

plt.rcParams.update({
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# kcal mol^-1 K^-1
R_KCAL = 0.00198720425864083

# 2.303 RT converts one pH unit into kcal/mol (~1.36 kcal/mol at 298 K).
TWO_303_RT = 2.303 * R_KCAL * TEMPERATURE_K

# Colour pairs: (free-energy line, occupancy curve).
# Each pair keeps a family identity -- acids (A-, HA) warm, bases (BH+, B)
# cool -- but uses two distinct, fully saturated hues so the two curves and
# their matching y-axes stay easy to tell apart.
RED = ("#c0392b", "#e07b00")   # acids: crimson free energy, orange occupancy
BLUE = ("#15398c", "#0f8f8f")  # bases: navy free energy, teal occupancy

# Species labels as mathtext.
SP_AMINUS = r"$\mathrm{A^-}$"
SP_HA = r"$\mathrm{HA}$"
SP_BHPLUS = r"$\mathrm{BH^+}$"
SP_B = r"$\mathrm{B}$"

# Reaction equilibrium schemes (mathtext), drawn near the top of each panel.
# Acids (A-, HA) share the HA/A- couple; bases (BH+, B) share the BH+/B couple.
RXN_ACID = r"$\mathrm{HA \rightleftharpoons A^- + H^+}$"
RXN_BASE = r"$\mathrm{BH^+ \rightleftharpoons B + H^+}$"

# Labels for the black zero line: it is the free energy of the plotted
# species' conjugate partner, i.e. the partner's own ΔG = 0.
ZERO_HA = r"$\mathbf{HA}_{\mathbf{\Delta G=0}}$"
ZERO_AMINUS = r"$\mathbf{A^-}_{\mathbf{\Delta G=0}}$"
ZERO_BHPLUS = r"$\mathbf{BH^+}_{\mathbf{\Delta G=0}}$"
ZERO_B = r"$\mathbf{B}_{\mathbf{\Delta G=0}}$"

# Governing equations per species (mathtext): free energy and probability.
# Drawn colour-coded in an in-panel legend by ``plot_panel``.
EQ_DG_AMINUS = r"$\Delta G_{\mathrm{A^-}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"
EQ_F_AMINUS = r"$f_{\mathrm{A^-}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"
EQ_DG_HA = r"$\Delta G_{\mathrm{HA}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"
EQ_F_HA = r"$f_{\mathrm{HA}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"
EQ_DG_BHPLUS = r"$\Delta G_{\mathrm{BH^+}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"
EQ_F_BHPLUS = r"$f_{\mathrm{BH^+}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"
EQ_DG_B = r"$\Delta G_{\mathrm{B}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"
EQ_F_B = r"$f_{\mathrm{B}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"


# ============================= Calculations ============================== #

def calculate_values(
    ph_min: float = PH_MIN,
    ph_max: float = PH_MAX,
    n_points: int = N_POINTS,
) -> pd.DataFrame:
    pH = np.linspace(ph_min, ph_max, n_points)

    # Relative free energies in kcal/mol.
    # Each expression is defined relative to the conjugate protonation state.
    dG_Aminus = TWO_303_RT * (PKA - pH)  # G(A-) - G(HA)
    dG_HA = TWO_303_RT * (pH - PKA)      # G(HA) - G(A-)

    dG_BHplus = TWO_303_RT * (pH - PKA)  # G(BH+) - G(B)
    dG_B = TWO_303_RT * (PKA - pH)       # G(B) - G(BH+)

    # Henderson-Hasselbalch fractional occupancies.
    fraction_Aminus = 1.0 / (1.0 + 10.0 ** (PKA - pH))
    fraction_HA = 1.0 / (1.0 + 10.0 ** (pH - PKA))

    fraction_BHplus = 1.0 / (1.0 + 10.0 ** (pH - PKA))
    fraction_B = 1.0 / (1.0 + 10.0 ** (PKA - pH))

    return pd.DataFrame(
        {
            "pH": pH,
            "dG_Aminus_relative_HA_kcal_per_mol": dG_Aminus,
            "fraction_Aminus": fraction_Aminus,
            "dG_HA_relative_Aminus_kcal_per_mol": dG_HA,
            "fraction_HA": fraction_HA,
            "dG_BHplus_relative_B_kcal_per_mol": dG_BHplus,
            "fraction_BHplus": fraction_BHplus,
            "dG_B_relative_BHplus_kcal_per_mol": dG_B,
            "fraction_B": fraction_B,
        }
    )


# =============================== Plotting ================================ #

def configure_panel(
    ax: plt.Axes,
    ax_fraction: plt.Axes,
    y_limit: float,
    pka_dx: float,
    pka_ha: str,
    colors: tuple[str, str],
    reference: str,
) -> None:
    dark, light = colors

    ax.set_xlim(PH_MIN, PH_MAX)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks(np.arange(int(PH_MIN), int(PH_MAX) + 1, 1))
    ax.set_xlabel("pH")
    ax.set_ylabel(r"Relative free energy, $\Delta G$ (kcal/mol)", color=dark)
    ax.grid(alpha=0.25)

    # Bold zero line: it is the conjugate partner's free energy (ΔG = 0),
    # labelled with a ZERO_* string below. Thinner than the 2.6 diagonal.
    ax.axhline(0.0, linewidth=ZERO_LINE_WIDTH, color="black", zorder=3)
    ax.axvline(PKA, linewidth=1.0, linestyle="--", color="black")

    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_fraction.set_ylabel("Probability", color=light)

    # Colour each y-axis to match its curve: left = free energy (dark),
    # right = occupancy (light), so it is clear which axis reads which.
    ax.tick_params(axis="y", colors=dark)
    ax.spines["left"].set_color(dark)
    ax_fraction.tick_params(axis="y", colors=light)
    ax_fraction.spines["right"].set_color(light)
    ax_fraction.spines["left"].set_visible(False)

    # Bold the y-axis tick labels on both axes for readability.
    for tick_label in ax.get_yticklabels() + ax_fraction.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.scatter(
        [PKA], [0.0],
        s=55, marker="s",
        facecolors="white", edgecolors="black",
        zorder=7,
    )

    ax.annotate(
        rf"$\mathrm{{p}}K_a={PKA:g}$",
        xy=(PKA, 0.0),
        xytext=(pka_dx, -16),
        textcoords="offset points",
        ha=pka_ha,
        fontsize=FS_PKA,
    )

    # Name the reference state on the black ΔG = 0 line: it is the free energy
    # of the conjugate partner the plotted species is measured against. Always
    # at the LEFT end, just above the line, in every panel -- beside the ΔG
    # axis it belongs to. Clear there for either slope: the diagonal is far
    # above or below the line at the left edge, the sigmoid sits on a plateau
    # at the top or bottom, the pKa label is near the pKa below the line, and
    # the equations sit at 0.28-0.36 / 0.64-0.72 of the panel height.
    ref_x = PH_MIN + 0.03 * (PH_MAX - PH_MIN)
    ax.annotate(
        reference,
        xy=(ref_x, 0.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left", va="bottom",
        color="black",
        fontsize=FS_REFERENCE,
        zorder=8,
    )


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dG_col: str,
    frac_col: str,
    colors: tuple[str, str],
    species: str,
    fe_xfrac: float,
    occ_frac: float,
    y_limit: float,
    panel_label: str,
    reaction: str,
    eq_dG: str,
    eq_frac: str,
    reference: str,
) -> dict:
    """Draw one species panel: free-energy line + probability sigmoid.

    ``panel_label`` (e.g. ``"A"``) is set as a bold left-aligned title above
    the subplot. ``reaction`` is the equilibrium scheme drawn near the top of
    the panel (always shown). ``eq_dG``/``eq_frac`` are the two governing
    equations, drawn colour-coded inside the panel (unless ``--no_legend``).
    ``reference`` is the label for the black zero line (a ``ZERO_*``
    constant, e.g. HA with subscript ΔG=0): the conjugate partner's free energy.

    ``fe_xfrac`` gives the x-position of the free-energy label as a fraction
    (0-1) of the plotted pH window, so it stays in-frame for any
    ``--ph-min``/``--ph-max`` choice. The probability label is anchored
    automatically at the plateau-side edge of the window (see below).

    The probability label is placed here; the free-energy label rides along
    the diagonal line and is added later (see ``place_line_labels``), once
    the final axes geometry is known so its rotation matches the slope.
    """
    dark, light = colors
    ax_fraction = ax.twinx()

    # Map the fractional free-energy label position onto the current pH window.
    fe_pH = PH_MIN + fe_xfrac * (PH_MAX - PH_MIN)

    ax.plot(df["pH"], df[dG_col], linewidth=2.6, color=dark, zorder=5)
    ax_fraction.plot(df["pH"], df[frac_col], linewidth=2.0, color=light, zorder=4)

    y0 = float(df[dG_col].iloc[0])
    y1 = float(df[dG_col].iloc[-1])

    # Place "pKa = 7" in the lower quadrant the diagonal does NOT cross:
    # lower-left for a descending line, lower-right for an ascending one.
    descending = y0 > y1
    pka_dx, pka_ha = (-10, "right") if descending else (10, "left")

    configure_panel(
        ax, ax_fraction, y_limit, pka_dx, pka_ha, colors,
        reference,
    )

    # Panel identifier (e.g. "A") above the subplot, pushed out to the far
    # top-left (into the margin) and set large and bold.
    ax.text(
        -0.13, 1.05, panel_label,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=FS_PANEL_LABEL, fontweight="bold",
    )

    # Reaction equilibrium scheme centred near the top of the panel. Always
    # drawn (independent of --no_legend); the top-centre stays clear of both
    # curves (the diagonal and sigmoid both pass through mid-height at the pKa).
    ax_fraction.text(
        0.5, 0.955, reaction, transform=ax_fraction.transAxes,
        ha="center", va="top", fontsize=FS_REACTION, zorder=9,
        bbox=EQ_BBOX,
    )

    # The two governing equations, colour-coded, drawn as plain text (not a
    # legend box) on the left side, mid-height: delta-G on top, f below. These
    # are the toggleable "legend" equations -- omitted when --no_legend (unlike
    # the reaction scheme above, which is always shown). The zero line
    # (mid-height, full width) and the diagonal force a direction-aware choice --
    # lower-left for a descending line (its upper-left is filled by the diagonal),
    # upper-left for an ascending one -- so they clear the zero line, the
    # diagonal, the sigmoid plateau, and the centred pKa label.
    if not NO_LEGEND:
        eq_y_dG, eq_y_f = (0.36, 0.28) if descending else (0.72, 0.64)
        ax_fraction.text(
            0.03, eq_y_dG, eq_dG, transform=ax_fraction.transAxes,
            ha="left", va="center", color=dark, fontsize=FS_EQUATION, zorder=9,
            bbox=EQ_BBOX,
        )
        ax_fraction.text(
            0.03, eq_y_f, eq_frac, transform=ax_fraction.transAxes,
            ha="left", va="center", color=light, fontsize=FS_EQUATION, zorder=9,
            bbox=EQ_BBOX,
        )

    # Occupancy label: anchored at the plateau-side edge of the window and
    # aligned so the text grows inward over the flat plateau -- never back
    # across the steep transition. This keeps it clear at any window width.
    rising = float(df[frac_col].iloc[-1]) > float(df[frac_col].iloc[0])
    margin = 0.03 * (PH_MAX - PH_MIN)
    if rising:                       # plateau at high pH -> anchor at right edge
        occ_x, occ_ha = PH_MAX - margin, "right"
    else:                            # plateau at low pH  -> anchor at left edge
        occ_x, occ_ha = PH_MIN + margin, "left"
    ax_fraction.text(
        occ_x, occ_frac, f"{species} probability",
        ha=occ_ha, va="center", color=dark, fontsize=FS_CURVE_LABEL,
    )

    # Endpoints of the (linear) free-energy line, for the rotated label.
    return {
        "ax": ax,
        "text": f"{species} free energy",
        "color": dark,
        "fe_pH": fe_pH,
        "y0": y0,
        "y1": y1,
    }


def place_line_labels(labels: list[dict]) -> None:
    """Add each free-energy label riding along (just above) its line.

    Called after ``tight_layout`` so the display rotation matches the line
    slope as actually rendered.
    """
    for entry in labels:
        ax = entry["ax"]
        y0, y1 = entry["y0"], entry["y1"]
        y_at = y0 + (y1 - y0) * (entry["fe_pH"] - PH_MIN) / (PH_MAX - PH_MIN)

        p0 = ax.transData.transform((PH_MIN, y0))
        p1 = ax.transData.transform((PH_MAX, y1))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

        ax.annotate(
            entry["text"],
            xy=(entry["fe_pH"], y_at),
            xytext=(0, 6),
            textcoords="offset points",
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            color=entry["color"],
            fontsize=FS_CURVE_LABEL,
        )


def plot_ionized_states(df: pd.DataFrame) -> plt.Figure:
    """Ionized species: deprotonated acid A- and protonated base BH+."""
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))

    labels = [
        plot_panel(
            axes[0], df,
            "dG_Aminus_relative_HA_kcal_per_mol", "fraction_Aminus",
            RED,
            SP_AMINUS,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="A",
            reaction=RXN_ACID,
            eq_dG=EQ_DG_AMINUS, eq_frac=EQ_F_AMINUS,
            reference=ZERO_HA,
        ),
        plot_panel(
            axes[1], df,
            "dG_BHplus_relative_B_kcal_per_mol", "fraction_BHplus",
            BLUE,
            SP_BHPLUS,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="B",
            reaction=RXN_BASE,
            eq_dG=EQ_DG_BHPLUS, eq_frac=EQ_F_BHPLUS,
            reference=ZERO_B,
        ),
    ]

    fig.tight_layout()
    place_line_labels(labels)
    return fig


def plot_neutral_states(df: pd.DataFrame) -> plt.Figure:
    """Neutral species: protonated acid HA and deprotonated base B."""
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))

    labels = [
        plot_panel(
            axes[0], df,
            "dG_HA_relative_Aminus_kcal_per_mol", "fraction_HA",
            RED,
            SP_HA,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="A",
            reaction=RXN_ACID,
            eq_dG=EQ_DG_HA, eq_frac=EQ_F_HA,
            reference=ZERO_AMINUS,
        ),
        plot_panel(
            axes[1], df,
            "dG_B_relative_BHplus_kcal_per_mol", "fraction_B",
            BLUE,
            SP_B,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="B",
            reaction=RXN_BASE,
            eq_dG=EQ_DG_B, eq_frac=EQ_F_B,
            reference=ZERO_BHPLUS,
        ),
    ]

    fig.tight_layout()
    place_line_labels(labels)
    return fig


def plot_all_states(df: pd.DataFrame) -> plt.Figure:
    """All four species on one 2x2 figure, panels A/B/C/D.

    A deprotonated acid A-    B protonated base BH+   (ionized states)
    C protonated acid HA      D deprotonated base B   (neutral states)
    """
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.6))

    labels = [
        plot_panel(
            axes[0, 0], df,
            "dG_Aminus_relative_HA_kcal_per_mol", "fraction_Aminus",
            RED,
            SP_AMINUS,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="A",
            reaction=RXN_ACID,
            eq_dG=EQ_DG_AMINUS, eq_frac=EQ_F_AMINUS,
            reference=ZERO_HA,
        ),
        plot_panel(
            axes[0, 1], df,
            "dG_BHplus_relative_B_kcal_per_mol", "fraction_BHplus",
            BLUE,
            SP_BHPLUS,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="B",
            reaction=RXN_BASE,
            eq_dG=EQ_DG_BHPLUS, eq_frac=EQ_F_BHPLUS,
            reference=ZERO_B,
        ),
        plot_panel(
            axes[1, 0], df,
            "dG_HA_relative_Aminus_kcal_per_mol", "fraction_HA",
            RED,
            SP_HA,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="C",
            reaction=RXN_ACID,
            eq_dG=EQ_DG_HA, eq_frac=EQ_F_HA,
            reference=ZERO_AMINUS,
        ),
        plot_panel(
            axes[1, 1], df,
            "dG_B_relative_BHplus_kcal_per_mol", "fraction_B",
            BLUE,
            SP_B,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
            panel_label="D",
            reaction=RXN_BASE,
            eq_dG=EQ_DG_B, eq_frac=EQ_F_B,
            reference=ZERO_BHPLUS,
        ),
    ]

    fig.tight_layout()
    place_line_labels(labels)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pH-dependent relative free energies and Henderson-Hasselbalch "
            "occupancies. The pKa and the plotted pH window are configurable; the "
            "exported CSV always spans the full pH 0-14 range."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # default: pKa = 7, plot pH 0-14\n"
            "  python3 plot_acid_base_with_equations.py\n"
            "\n"
            "  # different pKa\n"
            "  python3 plot_acid_base_with_equations.py --pka 4.5\n"
            "\n"
            "  # narrowed plot window (CSV still spans pH 0-14)\n"
            "  python3 plot_acid_base_with_equations.py --ph-min 4 --ph-max 10\n"
            "\n"
            "  # combine both\n"
            "  python3 plot_acid_base_with_equations.py --pka 9 --ph-min 6 --ph-max 12\n"
            "\n"
            "  # drop the equation legend from every figure\n"
            "  python3 plot_acid_base_with_equations.py --no_legend\n"
            "\n"
            "  # also open the figures in a window (default: save only)\n"
            "  python3 plot_acid_base_with_equations.py --show\n"
        ),
    )
    parser.add_argument(
        "--pka", type=float, default=PKA,
        help="pKa used for both the free-energy line and the occupancy sigmoid "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--ph-min", type=float, default=PH_MIN,
        help="Lower pH bound for the plots (default: %(default)s).",
    )
    parser.add_argument(
        "--ph-max", type=float, default=PH_MAX,
        help="Upper pH bound for the plots (default: %(default)s).",
    )
    parser.add_argument(
        "--no_legend", action="store_true",
        help="Omit the equation legend from every figure.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the figures in an interactive window after saving them "
             "(default: save to disk only, no window).",
    )
    return parser.parse_args()


def main() -> None:
    global PKA, PH_MIN, PH_MAX, NO_LEGEND

    args = parse_args()
    if args.ph_min >= args.ph_max:
        raise SystemExit(
            f"--ph-min ({args.ph_min}) must be less than --ph-max ({args.ph_max})."
        )
    PKA = args.pka
    PH_MIN, PH_MAX = args.ph_min, args.ph_max
    NO_LEGEND = args.no_legend

    output_directory = Path("plot_acid_base_with_equations")
    output_directory.mkdir(exist_ok=True)

    created: list[tuple[Path, str]] = []

    # The CSV always covers the full pH 0-14 range, independent of the plot window.
    csv_path = output_directory / "acid_base_values.csv"
    csv_df = calculate_values(0.0, 14.0, N_POINTS)
    csv_df.to_csv(csv_path, index=False)
    created.append((csv_path, f"values table, pH 0-14, {len(csv_df)} rows"))

    # Figures use the (possibly narrowed) plot window from --ph-min/--ph-max.
    df = calculate_values(PH_MIN, PH_MAX, N_POINTS)

    figures = (
        (plot_ionized_states, "figure_ionized_states_equations",
         "ionized states: A- (A), BH+ (B)"),
        (plot_neutral_states, "figure_neutral_states_equations",
         "neutral states: HA (A), B (B)"),
        (plot_all_states, "figure_all_states_equations",
         "all states, 2x2: A-, BH+, HA, B (A-D)"),
    )
    for plot_function, stem, what in figures:
        figure = plot_function(df)
        png_path = output_directory / f"{stem}.png"
        pdf_path = output_directory / f"{stem}.pdf"
        figure.savefig(png_path, dpi=DPI, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        created.append((png_path, f"{what}, {DPI} dpi"))
        created.append((pdf_path, what))

    # Summary of what was written and where.
    print(f"Wrote {len(created)} files to {output_directory.resolve()}/")
    print(
        f"  settings: pKa = {PKA:g}, plot pH {PH_MIN:g}-{PH_MAX:g}, "
        f"equations {'hidden (--no_legend)' if NO_LEGEND else 'shown'}"
    )
    width = max(len(path.name) for path, _ in created)
    for path, what in created:
        print(f"  {path.name:<{width}}  {what}")

    # Figures are always saved; a window opens only when asked for.
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
