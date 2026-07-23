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
from matplotlib.lines import Line2D


# ============================= User settings ============================= #

PKA = 7.0
TEMPERATURE_K = 298.15
PH_MIN = 0.0
PH_MAX = 14.0
N_POINTS = 1401
DPI = 300

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
) -> None:
    dark, light = colors

    ax.set_xlim(PH_MIN, PH_MAX)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks(np.arange(int(PH_MIN), int(PH_MAX) + 1, 1))
    ax.set_xlabel("pH")
    ax.set_ylabel(r"Relative free energy, $\Delta G$ (kcal/mol)", color=dark)
    ax.grid(alpha=0.25)

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.axvline(PKA, linewidth=1.0, linestyle="--", color="black")

    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_fraction.set_ylabel("Fractional occupancy", color=light)

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
        fontsize=11,
    )


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dG_col: str,
    frac_col: str,
    colors: tuple[str, str],
    title: str,
    species: str,
    fe_xfrac: float,
    occ_frac: float,
    y_limit: float,
) -> dict:
    """Draw one species panel: free-energy line + occupancy sigmoid.

    ``fe_xfrac`` gives the x-position of the free-energy label as a fraction
    (0-1) of the plotted pH window, so it stays in-frame for any
    ``--ph-min``/``--ph-max`` choice. The occupancy label is anchored
    automatically at the plateau-side edge of the window (see below).

    The occupancy label is placed here; the free-energy label rides along
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

    ax.set_title(title, fontsize=13)
    configure_panel(ax, ax_fraction, y_limit, pka_dx, pka_ha, colors)

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
        occ_x, occ_frac, f"{species} occupancy",
        ha=occ_ha, va="center", color=dark, fontsize=11.5,
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
            fontsize=11.5,
        )


def add_equation_legend(
    fig: plt.Figure,
    entries: list[tuple[str, str]],
) -> None:
    """Bottom legend collecting the governing equations (colour-coded)."""
    handles = [
        Line2D([0], [0], linewidth=2.6, color=color, label=label)
        for color, label in entries
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
        title=r"$2.303\,RT \approx 1.364$ kcal/mol at 298 K",
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
            r"A. Single Acid ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
            SP_AMINUS,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
        ),
        plot_panel(
            axes[1], df,
            "dG_BHplus_relative_B_kcal_per_mol", "fraction_BHplus",
            BLUE,
            r"B. Single Base ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
            SP_BHPLUS,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
        ),
    ]

    add_equation_legend(
        fig,
        [
            # Left column: acid equations.  Right column: base equations.
            (RED[0], r"$\Delta G_{\mathrm{A^-}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (RED[1], r"$f_{\mathrm{A^-}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
            (BLUE[0], r"$\Delta G_{\mathrm{BH^+}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (BLUE[1], r"$f_{\mathrm{BH^+}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
        ],
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
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
            r"A. Single Acid ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
            SP_HA,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
        ),
        plot_panel(
            axes[1], df,
            "dG_B_relative_BHplus_kcal_per_mol", "fraction_B",
            BLUE,
            r"B. Single Base ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
            SP_B,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
        ),
    ]

    add_equation_legend(
        fig,
        [
            # Left column: acid equations.  Right column: base equations.
            (RED[0], r"$\Delta G_{\mathrm{HA}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (RED[1], r"$f_{\mathrm{HA}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
            (BLUE[0], r"$\Delta G_{\mathrm{B}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (BLUE[1], r"$f_{\mathrm{B}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
        ],
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    place_line_labels(labels)
    return fig


def plot_all_states(df: pd.DataFrame) -> plt.Figure:
    """All four species on one 2x2 figure, panels A/B/C/D.

    A. deprotonated acid A-      B. protonated base BH+   (ionized states)
    C. protonated acid HA        D. deprotonated base B   (neutral states)
    """
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.6))

    labels = [
        plot_panel(
            axes[0, 0], df,
            "dG_Aminus_relative_HA_kcal_per_mol", "fraction_Aminus",
            RED,
            r"A. Single Acid, $\mathrm{A^-}$ ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
            SP_AMINUS,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
        ),
        plot_panel(
            axes[0, 1], df,
            "dG_BHplus_relative_B_kcal_per_mol", "fraction_BHplus",
            BLUE,
            r"B. Single Base, $\mathrm{BH^+}$ ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
            SP_BHPLUS,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
        ),
        plot_panel(
            axes[1, 0], df,
            "dG_HA_relative_Aminus_kcal_per_mol", "fraction_HA",
            RED,
            r"C. Single Acid, $\mathrm{HA}$ ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
            SP_HA,
            fe_xfrac=0.83, occ_frac=0.90,
            y_limit=y_limit,
        ),
        plot_panel(
            axes[1, 1], df,
            "dG_B_relative_BHplus_kcal_per_mol", "fraction_B",
            BLUE,
            r"D. Single Base, $\mathrm{B}$ ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
            SP_B,
            fe_xfrac=0.17, occ_frac=0.90,
            y_limit=y_limit,
        ),
    ]

    add_equation_legend(
        fig,
        [
            # Left column: all four acid equations (A- and HA).
            (RED[0], r"$\Delta G_{\mathrm{A^-}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (RED[1], r"$f_{\mathrm{A^-}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
            (RED[0], r"$\Delta G_{\mathrm{HA}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (RED[1], r"$f_{\mathrm{HA}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
            # Right column: all four base equations (BH+ and B).
            (BLUE[0], r"$\Delta G_{\mathrm{BH^+}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (BLUE[1], r"$f_{\mathrm{BH^+}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
            (BLUE[0], r"$\Delta G_{\mathrm{B}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (BLUE[1], r"$f_{\mathrm{B}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
        ],
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
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
    return parser.parse_args()


def main() -> None:
    global PKA, PH_MIN, PH_MAX

    args = parse_args()
    if args.ph_min >= args.ph_max:
        raise SystemExit(
            f"--ph-min ({args.ph_min}) must be less than --ph-max ({args.ph_max})."
        )
    PKA = args.pka
    PH_MIN, PH_MAX = args.ph_min, args.ph_max

    output_directory = Path("plot_acid_base_with_equations")
    output_directory.mkdir(exist_ok=True)

    # The CSV always covers the full pH 0-14 range, independent of the plot window.
    csv_df = calculate_values(0.0, 14.0, N_POINTS)
    csv_df.to_csv(output_directory / "acid_base_values.csv", index=False)

    # Figures use the (possibly narrowed) plot window from --ph-min/--ph-max.
    df = calculate_values(PH_MIN, PH_MAX, N_POINTS)

    ionized_figure = plot_ionized_states(df)
    ionized_figure.savefig(
        output_directory / "figure_ionized_states_equations.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    ionized_figure.savefig(
        output_directory / "figure_ionized_states_equations.pdf",
        bbox_inches="tight",
    )

    neutral_figure = plot_neutral_states(df)
    neutral_figure.savefig(
        output_directory / "figure_neutral_states_equations.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    neutral_figure.savefig(
        output_directory / "figure_neutral_states_equations.pdf",
        bbox_inches="tight",
    )

    all_figure = plot_all_states(df)
    all_figure.savefig(
        output_directory / "figure_all_states_equations.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    all_figure.savefig(
        output_directory / "figure_all_states_equations.pdf",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
