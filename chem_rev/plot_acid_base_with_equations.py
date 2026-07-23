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

# Colour pairs: (dark = free-energy line, light = occupancy curve).
# Acids (A-, HA) use a red scheme; bases (BH+, B) use a blue scheme.
RED = ("#c0392b", "#e79a90")   # acids: A- and HA
BLUE = ("#1f4e9c", "#7ea8dd")  # bases: BH+ and B

# Species labels as mathtext.
SP_AMINUS = r"$\mathrm{A^-}$"
SP_HA = r"$\mathrm{HA}$"
SP_BHPLUS = r"$\mathrm{BH^+}$"
SP_B = r"$\mathrm{B}$"


# ============================= Calculations ============================== #

def calculate_values() -> pd.DataFrame:
    pH = np.linspace(PH_MIN, PH_MAX, N_POINTS)

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

def configure_panel(ax: plt.Axes, ax_fraction: plt.Axes, y_limit: float) -> None:
    ax.set_xlim(PH_MIN, PH_MAX)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks(np.arange(int(PH_MIN), int(PH_MAX) + 1, 1))
    ax.set_xlabel("pH")
    ax.set_ylabel(r"Relative free energy, $\Delta G$ (kcal/mol)")
    ax.grid(alpha=0.25)

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.axvline(PKA, linewidth=1.0, linestyle="--", color="black")

    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_fraction.set_ylabel("Fractional occupancy")

    ax.scatter(
        [PKA], [0.0],
        s=55, marker="s",
        facecolors="white", edgecolors="black",
        zorder=7,
    )

    ax.annotate(
        rf"$\mathrm{{p}}K_a={PKA:g}$",
        xy=(PKA, 0.0),
        xytext=(10, -16),
        textcoords="offset points",
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
    fe_xy: tuple[float, float],
    occ_xy: tuple[float, float],
    y_limit: float,
) -> None:
    """Draw one species panel: free-energy line + occupancy sigmoid."""
    dark, light = colors
    ax_fraction = ax.twinx()

    ax.plot(df["pH"], df[dG_col], linewidth=2.6, color=dark, zorder=5)
    ax_fraction.plot(df["pH"], df[frac_col], linewidth=2.0, color=light, zorder=4)

    ax.set_title(title, fontsize=13)
    configure_panel(ax, ax_fraction, y_limit)

    # Inline species labels placed near each curve.
    ax.text(
        fe_xy[0], fe_xy[1], f"{species} free energy",
        transform=ax.transAxes, color=dark,
        fontsize=12, fontweight="bold", va="center",
    )
    ax_fraction.text(
        occ_xy[0], occ_xy[1], f"{species} occupancy",
        transform=ax_fraction.transAxes, color=dark,
        fontsize=12, fontweight="bold", va="center",
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
        title=r"$2.303\,RT \approx 1.36$ kcal/mol at 298 K",
    )


def plot_ionized_states(df: pd.DataFrame) -> plt.Figure:
    """Ionized species: deprotonated acid A- and protonated base BH+."""
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))

    plot_panel(
        axes[0], df,
        "dG_Aminus_relative_HA_kcal_per_mol", "fraction_Aminus",
        RED,
        r"A. Single acid ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
        SP_AMINUS,
        fe_xy=(0.07, 0.94), occ_xy=(0.42, 0.90),
        y_limit=y_limit,
    )
    plot_panel(
        axes[1], df,
        "dG_BHplus_relative_B_kcal_per_mol", "fraction_BHplus",
        BLUE,
        r"B. Single base ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
        SP_BHPLUS,
        fe_xy=(0.28, 0.10), occ_xy=(0.04, 0.90),
        y_limit=y_limit,
    )

    add_equation_legend(
        fig,
        [
            (RED[0], r"$\Delta G_{\mathrm{A^-}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (BLUE[0], r"$\Delta G_{\mathrm{BH^+}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (RED[1], r"$f_{\mathrm{A^-}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
            (BLUE[1], r"$f_{\mathrm{BH^+}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
        ],
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return fig


def plot_neutral_states(df: pd.DataFrame) -> plt.Figure:
    """Neutral species: protonated acid HA and deprotonated base B."""
    y_limit = 1.08 * TWO_303_RT * max(abs(PH_MIN - PKA), abs(PH_MAX - PKA))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))

    plot_panel(
        axes[0], df,
        "dG_HA_relative_Aminus_kcal_per_mol", "fraction_HA",
        RED,
        r"A. Single acid ($\mathrm{HA \rightleftharpoons A^- + H^+}$)",
        SP_HA,
        fe_xy=(0.20, 0.10), occ_xy=(0.04, 0.90),
        y_limit=y_limit,
    )
    plot_panel(
        axes[1], df,
        "dG_B_relative_BHplus_kcal_per_mol", "fraction_B",
        BLUE,
        r"B. Single base ($\mathrm{BH^+ \rightleftharpoons B + H^+}$)",
        SP_B,
        fe_xy=(0.07, 0.94), occ_xy=(0.62, 0.90),
        y_limit=y_limit,
    )

    add_equation_legend(
        fig,
        [
            (RED[0], r"$\Delta G_{\mathrm{HA}}=2.303RT(\mathrm{pH}-\mathrm{p}K_a)$"),
            (BLUE[0], r"$\Delta G_{\mathrm{B}}=2.303RT(\mathrm{p}K_a-\mathrm{pH})$"),
            (RED[1], r"$f_{\mathrm{HA}}=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a)}\right]^{-1}$"),
            (BLUE[1], r"$f_{\mathrm{B}}=\left[1+10^{(\mathrm{p}K_a-\mathrm{pH})}\right]^{-1}$"),
        ],
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return fig


def main() -> None:
    output_directory = Path("plot_acid_base_with_equations")
    output_directory.mkdir(exist_ok=True)
    df = calculate_values()

    df.to_csv(output_directory / "acid_base_values.csv", index=False)

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

    plt.show()


if __name__ == "__main__":
    main()
