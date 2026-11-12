#!/usr/bin/env python3
"""
Proton-loading-site (PLS) uptake from Henderson-Hasselbalch titration curves.

Regenerates -- analytically, on a full pKa x pH grid -- the model behind the
``PLS-HH`` sheet / ``dpK 7-8 and PLS uptake`` chart of ``aa-freq.xlsx``, which
plots three curves against pH:

    f_loaded    protonation of the site in its *loaded*   (high-pKa) state
    f_unloaded  protonation of the site in its *unloaded* (low-pKa)  state
    delta_f     the difference, f_unloaded - f_loaded

``delta_f`` follows the spreadsheet's sign convention (column ``L = H - I``, the
low-pKa state minus the high-pKa one), so it is *negative*: it is the protonation
lost on unloading. Its magnitude is the protons taken up per cycle.

delta_f is a bell curve, extremal midway between the two pKas at

    delta_f_min = -(10^(dpKa/2) - 1) / (10^(dpKa/2) + 1)

so a full proton per cycle requires a *large enough pKa swing*, not merely a
shifted pKa. The figure shows three ways of opening that swing up from an
anchor pKa, each stepped until |delta_f| reaches 1:

    (A)  loaded state fixed at the anchor; the unloaded pKa is driven DOWN
    (B)  unloaded state fixed at the anchor; the loaded pKa is driven UP
    (C)  both states moved together, symmetrically about the anchor

Within a panel every pKa swing is numbered, and its number is repeated on each of
the curves it produces -- loaded state, unloaded state, difference -- so the three
can be read as one set.

All outputs are written into the ``plot_pls_uptake/`` directory (--outdir):
    figure_pls_uptake.png / .pdf            the three panels together
    figure_pls_uptake_A/_B/_C.png / .pdf    each panel again, on its own
    henderson_hasselbalch.csv   full pKa 0-14 x pH 0-14 grid
    pls_uptake_curves.csv       the three panels' plotted curves
    pls_uptake_maxima.csv       extremum of every difference curve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================= User settings ============================= #

# Anchor pKa: the fixed state in panels (A)/(B), the centre of symmetry in (C).
# Overridable with --anchor. 7.0 sits at the centre of the spreadsheet's pKa 6
# (unloaded) / 8 (loaded) pair, which panel (C) reproduces at dpKa = 2.
PKA_ANCHOR = 7.0

# The pKa ladder is grown in steps of DELTA_STEP until the uptake curve reaches
# UPTAKE_TARGET (dpKa = 6 gives 0.998, i.e. one proton to within 0.2%).
DELTA_STEP = 1.0
UPTAKE_TARGET = 0.995

# Absolute |delta_f| levels at which the width of the uptake window is reported
# (in pH units), alongside the width at half of each curve's own extremum.
WIDTH_LEVELS = (0.5, 0.9)

TEMPERATURE_K = 298.15
PH_MIN = 0.0
PH_MAX = 14.0
N_POINTS = 1401          # plotted resolution over the pH window (0.01 pH units)

# Exported Henderson-Hasselbalch grid: pKa 0-14 x pH 0-14, spreadsheet spacing.
PKA_GRID_MIN = 0.0
PKA_GRID_MAX = 14.0
GRID_STEP = 0.1

DPI = 300

# ------------------------------- Type sizes ------------------------------ #
# All in-panel type is sized from this one block (and the rcParams below it, for
# the axis and tick labels), so the whole figure family scales from here rather
# than from the call sites. Sized for a figure reduced to journal column width.
FS_PANEL_LABEL = 26      # the bold A/B/C identifier
FS_TITLE = 15
FS_REACTION = 15         # the reaction scheme, in the title or inside the panel
FS_RELEASE_LABEL = 14    # the in-panel name of the lower half
FS_EQUATION = 15         # the equation block above the combined figure
FS_EQUATION_SINGLE = 13  # ... and above a standalone panel, which is narrower
FS_ANNOTATION = 13       # delta_P_min at the widest swing's extremum
FS_LEGEND = 11
FS_BADGE = 12            # difference badges: one character, in a circle
FS_BADGE_ROW = 10.5      # state badges: two characters, in a pill

plt.rcParams.update({
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# When True (--no_legend), both the equation block above the figure and the curve
# legend below each panel are omitted. The reaction scheme and the panel titles
# are descriptions of the system rather than legends, and are always drawn.
NO_LEGEND = False

# kcal mol^-1 K^-1
R_KCAL = 0.00198720425864083

# 2.303 RT converts one pH (or pKa) unit into kcal/mol (~1.36 kcal/mol at 298 K).
TWO_303_RT = 2.303 * R_KCAL * TEMPERATURE_K

# Hue carries state identity: the loaded state is red, the unloaded state blue.
# Within each family, a single-hue sequential ramp carries the *magnitude* of the
# pKa swing (light = small swing, dark = large), so the same dpKa is the same
# shade everywhere. The difference of the two states gets green -- a third hue,
# well clear of both state families -- on the same lightness steps.
RAMP_LOADED = plt.cm.Reds
RAMP_UNLOADED = plt.cm.Blues
RAMP_DIFFERENCE = plt.cm.Greens
RAMP_SPAN = (0.40, 0.95)

# The pinned anchor curve, drawn thick in the saturated project hue of its family
# (see plot_acid_base_with_equations.py) so it reads as one fixed reference
# rather than as a member of the ramp.
CRIMSON = "#c0392b"  # loaded state held fixed  (A)
NAVY = "#15398c"     # unloaded state held fixed (B)

# Numbered badges tie a panel's curves together: the loaded state, the unloaded
# state and the difference they produce all carry the *same* number, so one pKa
# swing can be traced across the three families without reading the legend (the
# number also opens that swing's legend entry). All badges share one neutral
# light-blue bubble -- the number is a grouping marker, deliberately not a fourth
# colour channel on top of the red/blue/green hues.
BADGE_FACE = "#cfe8f7"
BADGE_EDGE = "#2c6d92"
BADGE_TEXT = "#123c52"
BADGE_FONTSIZE = FS_BADGE
# Row badges carry two characters, so they are set a touch smaller: it keeps
# their footprint close to the single-character circles', and keeps the row
# readable in panel (C), where both families share it at half the pKa step.
BADGE_ROW_FONTSIZE = FS_BADGE_ROW
# The badges are aligned rather than scattered: the titration curves are badged
# along one horizontal line, at f = 0.5, so each badge sits on its own curve's
# pKa and the row reads off the pKa ladder; the difference curves are badged
# along one vertical line (see difference_column). Badges that would touch are
# spread apart *along* their shared line, so the alignment always holds.
#
# A state badge names its family as well as its swing -- 3U is the unloaded state
# of swing 3, 3L its loaded state -- because the two families share the row and,
# in panel (C), both are present. The difference badge is the bare number: it is
# the swing itself, 3U and 3L together. State badges are therefore two characters
# and sit in a pill; difference badges stay single-character circles.
BADGE_ROW_F = 0.5
# When the pKa ladder is finer than the badges are wide -- panel (C), where the
# two families step by half of --delta-step -- the row splits into two levels,
# this far apart, and alternates between them. A badge then keeps its curve's
# exact pKa (the row's whole point) instead of being slid along the row away
# from the curve it names; the sigmoid is steep at its pKa, so a badge half a
# level above or below f = 0.5 still sits on it.
BADGE_ROW_STAGGER = 0.16
BADGE_ROW_BOX = "round,pad=0.28"
BADGE_COLUMN_BOX = "circle,pad=0.3"
BADGE_MIN_GAP_PH = 0.85   # clearance along the row,    in pH units
BADGE_MIN_GAP_F = 0.14    # clearance along the column, in units of f
BADGE_COLUMN_TOP = -0.03  # the column stays wholly inside the difference half
BADGE_MARGIN = 0.5        # pH units of clearance kept from the axes edges

# The panel's y range: P from 0 to 1 above, delta_P from 0 to -1 below, plus
# headroom for the reaction scheme above and the annotated extremum below. Also
# what bounds the badge column (see fitted_gap).
PANEL_YLIM = (-1.18, 1.12)

# The difference curves' half of the shared y-axis: ticks and axis label on the
# left, below the zero line (see configure_panel), in the mid shade of their ramp.
DIFFERENCE_AXIS = "#1b7a3d"

# Both y-axis labels sit on the left, one per half, so they are placed by hand at
# this x (axes fraction) to keep them in line with each other -- see configure_panel.
YLABEL_X = -0.12

# Extremum markers on the difference curves. Deliberately outside the green ramp:
# a marker in its own curve's shade cannot be picked out from the dashes it sits on.
EXTREMUM_COLOR = "black"

# When True (--no_title), the panel titles are dropped and the reaction scheme --
# which is their first line -- is drawn inside the panel instead. The scheme is a
# description of the system and is never dropped, only relocated; the titles'
# second line (which pKa moves) is, and belongs in the figure caption.
NO_TITLE = False

# Reaction scheme (mathtext), always drawn: as the title's first line, or inside
# the panel when --no_title. Inside, it sits top centre, in the band above P = 1
# that no curve can reach. Written in the release direction (the loaded site on
# the left), to match the sign convention of the difference curves.
RXN_PLS = r"$\mathrm{PLS\!-\!H} \rightleftharpoons \mathrm{PLS} + \mathrm{H^+}$"
REACTION_Y = 1.06
REACTION_FONTSIZE = FS_REACTION

# The lower half's quantity is named twice: once inside the panel -- centred,
# below the zero line -- mirroring the reaction scheme above the upper half, so
# each half of the shared axis is named where it starts; and once as the green
# axis label alongside its own ticks on the left.
RELEASE_LABEL = "Proton Release\n" + r"Probability, $\Delta P$"
RELEASE_LABEL_Y = -0.07
RELEASE_LABEL_FONTSIZE = FS_RELEASE_LABEL
# Half the label's width, as a fraction of the plotted pH window: the band the
# badge column is kept out of (difference_column). Measured from the rendered
# label -- 0.161 of the panel on the triptych, less on the narrower standalone
# page -- plus clearance for the badge bubbles themselves.
RELEASE_LABEL_HALF_WIDTH = 0.18

# Governing equations, drawn once above the figure unless --no_legend.
EQ_F = (
    r"$P_i=\left[1+10^{(\mathrm{pH}-\mathrm{p}K_a^{\,i})}\right]^{-1}$"
)
EQ_UPTAKE = r"$\Delta P=P_{\mathrm{unloaded}}-P_{\mathrm{loaded}}$"
EQ_MAX = (
    r"$\Delta P_{\mathrm{min}}=-\dfrac{10^{\Delta \mathrm{p}K_a/2}-1}"
    r"{10^{\Delta \mathrm{p}K_a/2}+1}$ at "
    r"$\mathrm{pH}=\frac{1}{2}\left(\mathrm{p}K_a^{\,\mathrm{loaded}}"
    r"+\mathrm{p}K_a^{\,\mathrm{unloaded}}\right)$"
)


# ============================= Calculations ============================== #

def fraction_protonated(pH: np.ndarray | float, pka: float) -> np.ndarray | float:
    """Henderson-Hasselbalch protonated fraction of a site of given pKa."""
    return 1.0 / (1.0 + 10.0 ** (pH - pka))


def uptake_maximum(delta_pka: float) -> float:
    """Protons taken up per cycle for a pKa swing of ``delta_pka``: |delta_f|.

    The difference of two Henderson-Hasselbalch curves separated by dpKa is
    extremal midway between them, with magnitude (r - 1)/(r + 1) where
    r = 10^(dpKa/2). Returned positive; ``delta_f`` itself is its negative.
    """
    ratio = 10.0 ** (delta_pka / 2.0)
    return (ratio - 1.0) / (ratio + 1.0)


def delta_f_width(delta_pka: float, level: float) -> float:
    """pH width of the |delta_f| bell at the height ``level``; NaN if unreachable.

    With x = pH - midpoint and a = 10^(dpKa/2), the difference of the two
    Henderson-Hasselbalch curves is

        |delta_f|(x) = (a^2 - 1) / [2a cosh(x ln10) + 1 + a^2]

    which inverts in closed form: the two pH values at which the bell reaches
    ``level`` are symmetric about the midpoint, and their separation is

        width = 2 arccosh(C) / ln10,
        C = [(a^2 - 1)/level - (1 + a^2)] / 2a.

    C < 1 means the bell never gets that high -- the pKa swing is too small --
    and the width is undefined. C = 1 (level = |delta_f|_min) gives width 0.

    This is the answer to "how many pH units wide is the window in which the
    site actually cycles the proton": it tends to dpKa itself for large swings,
    but collapses for small ones (dpKa = 1 holds half a proton over 0.4 pH units).
    """
    if not 0.0 < level < 1.0:
        return float("nan")
    a_squared = 10.0 ** delta_pka
    a = 10.0 ** (delta_pka / 2.0)
    c = ((a_squared - 1.0) / level - (1.0 + a_squared)) / (2.0 * a)
    if c < 1.0:
        return float("nan")
    return 2.0 * float(np.arccosh(c)) / np.log(10.0)


def henderson_hasselbalch_grid(
    pka_min: float = PKA_GRID_MIN,
    pka_max: float = PKA_GRID_MAX,
    ph_min: float = 0.0,
    ph_max: float = 14.0,
    step: float = GRID_STEP,
) -> pd.DataFrame:
    """Every pKa from 0-14 titrated over pH 0-14, in tidy (long) form.

    One row per (pKa, pH) pair: the protonated and deprotonated fractions and
    the corresponding relative free energy of the protonated state.
    """
    n_pka = int(round((pka_max - pka_min) / step)) + 1
    n_ph = int(round((ph_max - ph_min) / step)) + 1
    pka_values = np.round(pka_min + step * np.arange(n_pka), 10)
    ph_values = np.round(ph_min + step * np.arange(n_ph), 10)

    pka_column, ph_column = (a.ravel() for a in np.meshgrid(pka_values, ph_values, indexing="ij"))
    protonated = fraction_protonated(ph_column, pka_column)

    return pd.DataFrame(
        {
            "pKa": pka_column,
            "pH": ph_column,
            "fraction_protonated": protonated,
            "fraction_deprotonated": 1.0 - protonated,
            # G(protonated) - G(deprotonated), the sign convention of the
            # companion script's dG_HA.
            "dG_protonated_relative_deprotonated_kcal_per_mol":
                TWO_303_RT * (ph_column - pka_column),
        }
    )


def delta_ladder(
    target: float = UPTAKE_TARGET,
    step: float = DELTA_STEP,
) -> list[float]:
    """pKa swings 1, 2, 3, ... up to the first one whose uptake reaches ``target``."""
    deltas: list[float] = []
    delta = step
    while True:
        deltas.append(delta)
        if uptake_maximum(delta) >= target or delta >= PKA_GRID_MAX:
            return deltas
        delta += step


# Each panel maps a pKa swing onto a (loaded, unloaded) pKa pair, and names
# which of the two is held fixed.
def _panel_a(delta: float, anchor: float) -> tuple[float, float]:
    """Loaded state pinned at the anchor; unloaded pKa driven down."""
    return anchor, anchor - delta


def _panel_b(delta: float, anchor: float) -> tuple[float, float]:
    """Unloaded state pinned at the anchor; loaded pKa driven up."""
    return anchor + delta, anchor


def _panel_c(delta: float, anchor: float) -> tuple[float, float]:
    """Both states moved together, symmetrically about the anchor."""
    return anchor + delta / 2.0, anchor - delta / 2.0


PANELS = (
    {
        "key": "A",
        "pair": _panel_a,
        "fixed": "loaded",
        "condition": (
            r"loaded state fixed at p$K_a$ {anchor:g}, unloaded p$K_a$ lowered"
        ),
    },
    {
        "key": "B",
        "pair": _panel_b,
        "fixed": "unloaded",
        "condition": (
            r"unloaded state fixed at p$K_a$ {anchor:g}, loaded p$K_a$ raised"
        ),
    },
    {
        "key": "C",
        "pair": _panel_c,
        "fixed": None,
        "condition": (
            r"both states moved, symmetric about p$K_a$ {anchor:g}"
        ),
    },
)


def panel_curves(
    panel: dict,
    deltas: list[float],
    anchor: float,
    pH: np.ndarray,
) -> list[dict]:
    """Build one entry per pKa swing: the two titration curves and their difference.

    Swings whose pKa pair would fall outside the exported pKa 0-14 grid are
    dropped, so every plotted curve is backed by a row of the CSV.
    """
    entries = []
    for delta in deltas:
        pka_loaded, pka_unloaded = panel["pair"](delta, anchor)
        if not (PKA_GRID_MIN <= pka_unloaded and pka_loaded <= PKA_GRID_MAX):
            continue

        f_loaded = fraction_protonated(pH, pka_loaded)
        f_unloaded = fraction_protonated(pH, pka_unloaded)
        entries.append(
            {
                "delta": delta,
                "pka_loaded": pka_loaded,
                "pka_unloaded": pka_unloaded,
                "f_loaded": f_loaded,
                "f_unloaded": f_unloaded,
                # Spreadsheet sign convention (L = H - I): negative.
                "delta_f": f_unloaded - f_loaded,
                "ph_at_extremum": 0.5 * (pka_loaded + pka_unloaded),
                "delta_f_min": -uptake_maximum(delta),
                "protons_per_cycle": uptake_maximum(delta),
                # How wide, in pH units, the uptake window is: at half of this
                # curve's own extremum, and at fixed absolute |delta_f| levels.
                "fwhm_pH": delta_f_width(delta, uptake_maximum(delta) / 2.0),
                "widths_pH": {
                    level: delta_f_width(delta, level) for level in WIDTH_LEVELS
                },
            }
        )
    return entries


# =============================== Plotting ================================ #

def declump(positions: list[float], min_gap: float) -> list[float]:
    """Spread points along a line until no two are closer than ``min_gap``.

    Order is preserved and the result is re-centred on the original span, so a
    set that is already clear comes back untouched and a crowded one opens
    symmetrically rather than drifting off one end.
    """
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    spread = [positions[index] for index in order]
    for k in range(1, len(spread)):
        spread[k] = max(spread[k], spread[k - 1] + min_gap)

    shift = 0.5 * (
        (positions[order[0]] + positions[order[-1]]) - (spread[0] + spread[-1])
    )
    aligned = [0.0] * len(positions)
    for k, index in enumerate(order):
        aligned[index] = spread[k] + shift
    return aligned


def fitted_gap(count: int, span: float, min_gap: float) -> float:
    """``min_gap``, or as much of it as fits when a line has too many badges.

    A long pKa ladder (say --delta-step 0.25) can ask for more room than the row
    or the column has; without this the spread would run off the panel. Crowded
    badges are still better than badges outside the axes.
    """
    if count < 2:
        return min_gap
    return min(min_gap, span / (count - 1))


def stagger_row(positions: list[float], min_gap: float) -> list[int]:
    """Sort a row of badges into two levels, so each can stay at its own pKa.

    Level 0 unless the badge would come within ``min_gap`` of the last one placed
    there, in which case level 1 -- and so on, alternating along the row. Sliding
    badges apart instead (``declump``) walks them away from the curves they name,
    which is what the row is for; alternating buys the same room for nothing but
    a small vertical offset. All zeros when the row is already clear, so a panel
    whose pKas are far enough apart keeps its single row.
    """
    levels = [0] * len(positions)
    last = [float("-inf"), float("-inf")]
    for index in sorted(range(len(positions)), key=lambda i: positions[i]):
        position = positions[index]
        if position - last[0] >= min_gap:
            level = 0
        elif position - last[1] >= min_gap:
            level = 1
        else:
            # Both are crowded: take the roomier one and let declump finish.
            level = 0 if position - last[0] >= position - last[1] else 1
        last[level] = position
        levels[index] = level
    return levels


def difference_column(entries: list[dict], pH: np.ndarray) -> tuple[float, list[float]]:
    """The pH at which the difference curves are furthest apart, and their values.

    All the difference badges go on this one vertical line, so it is chosen to
    maximise the *smallest* gap between neighbouring curves there -- which lands
    it on the flank of the bells, where they fan out, rather than at their
    extrema, where the deep ones are all pressed against -1.
    """
    stack = np.vstack([entry["delta_f"] for entry in entries])
    if stack.shape[0] < 2:
        index = int(np.argmin(stack[0]))
        return float(pH[index]), [float(stack[0, index])]

    smallest_gap = np.diff(np.sort(stack, axis=0), axis=0).min(axis=0)
    inside = (pH >= PH_MIN + BADGE_MARGIN) & (pH <= PH_MAX - BADGE_MARGIN)
    # The centred "Proton Release Probability" label occupies the top of the
    # difference half, where the shallowest badges sit; the column keeps clear of
    # the band it spans rather than running underneath it.
    half_width = RELEASE_LABEL_HALF_WIDTH * (PH_MAX - PH_MIN)
    midpoint = 0.5 * (PH_MIN + PH_MAX)
    inside &= np.abs(pH - midpoint) > half_width
    index = int(np.argmax(np.where(inside, smallest_gap, -np.inf)))
    return float(pH[index]), [float(value) for value in stack[:, index]]


def draw_badge(
    ax: plt.Axes,
    position: tuple[float, float],
    label: str,
    boxstyle: str,
    fontsize: float = BADGE_FONTSIZE,
) -> None:
    ax.text(
        position[0], position[1], label,
        ha="center", va="center", zorder=10,
        fontsize=fontsize, fontweight="bold", color=BADGE_TEXT,
        bbox=dict(
            boxstyle=boxstyle, facecolor=BADGE_FACE,
            edgecolor=BADGE_EDGE, linewidth=0.8,
        ),
    )


def draw_badges(
    ax: plt.Axes,
    panel: dict,
    entries: list[dict],
    column_ph: float,
    column_values: list[float],
) -> None:
    """Number every pKa swing, once on each of the curves it produces.

    Two aligned sets: the titration curves are badged along the horizontal
    f = 0.5 line (each badge therefore marking its curve's pKa, and naming its
    family: 3U unloaded, 3L loaded), the difference curves along one vertical
    line, as the bare swing number. The pinned state is one curve shared by every
    swing, so it carries no badge; only the states that move are badged.
    """
    row: list[tuple[float, str]] = []
    for number, entry in enumerate(entries, start=1):
        if panel["fixed"] != "loaded":
            row.append((entry["pka_loaded"], f"{number}L"))
        if panel["fixed"] != "unloaded":
            row.append((entry["pka_unloaded"], f"{number}U"))

    # A state is badged at its own pKa, so a state whose pKa falls outside the
    # plotted pH window cannot be: its half-height point is not in view. Its curve
    # is still drawn (and still named in the legend); only the badge is dropped,
    # rather than being pushed into the margin on top of the axis labels.
    row = [
        (pka, label) for pka, label in row
        if PH_MIN + BADGE_MARGIN <= pka <= PH_MAX - BADGE_MARGIN
    ]

    # Two levels where the ladder is finer than the badges are wide, one where it
    # is not; declump then only has to settle what is still crowded within a level.
    pkas = [pka for pka, _ in row]
    levels = stagger_row(pkas, BADGE_MIN_GAP_PH)
    offsets = (
        (0.5 * BADGE_ROW_STAGGER, -0.5 * BADGE_ROW_STAGGER) if any(levels)
        else (0.0, 0.0)
    )
    for level in (0, 1):
        members = [k for k, value in enumerate(levels) if value == level]
        if not members:          # the usual case for level 1: one row was enough
            continue
        aligned = declump(
            [pkas[k] for k in members],
            fitted_gap(
                len(members),
                (PH_MAX - PH_MIN) - 2 * BADGE_MARGIN,
                BADGE_MIN_GAP_PH,
            ),
        )
        for x, k in zip(aligned, members):
            draw_badge(
                ax, (x, BADGE_ROW_F + offsets[level]),
                row[k][1], BADGE_ROW_BOX, BADGE_ROW_FONTSIZE,
            )

    column = declump(
        column_values,
        fitted_gap(
            len(column_values),
            BADGE_COLUMN_TOP - PANEL_YLIM[0],
            BADGE_MIN_GAP_F,
        ),
    )
    # Where the shallowest curves are almost flat, spreading them apart would lift
    # the top of the column across the zero line; slide the whole column down
    # instead, so a difference badge never strays into the f half of the panel.
    overshoot = max(column) - BADGE_COLUMN_TOP
    if overshoot > 0.0:
        column = [y - overshoot for y in column]
    for number, y in enumerate(column, start=1):
        draw_badge(ax, (column_ph, y), str(number), BADGE_COLUMN_BOX)


def draw_reaction_scheme(ax: plt.Axes) -> None:
    """Draw the reaction scheme inside the panel, for when the title is dropped.

    Top centre, in the headroom above P = 1 that the ylim leaves clear: no curve
    can enter it -- a protonation probability cannot exceed 1 and a difference
    cannot be positive -- so the placement holds whatever the pKa ladder or pH
    window.
    """
    ax.text(
        0.5 * (PH_MIN + PH_MAX), REACTION_Y, RXN_PLS,
        ha="center", va="center", fontsize=REACTION_FONTSIZE, zorder=9,
        # Opaque enough to mask the anchor line, which runs behind it.
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
    )


def configure_panel(
    ax: plt.Axes, anchor: float, panel_label: str | None, condition: str
) -> None:
    ax.set_xlim(PH_MIN, PH_MAX)
    # The titration curves occupy 0 to 1, the (negative) difference curves 0 to
    # -1; the extra headroom is for the annotated extremum of the widest swing.
    ax.set_ylim(*PANEL_YLIM)
    ax.set_xticks(np.arange(int(PH_MIN), int(PH_MAX) + 1, 1))
    ax.set_xlabel("pH")
    ax.grid(alpha=0.25)

    # One axis carries two quantities: P in the upper half, delta_P in the lower.
    # Both are labelled on the *left*, each alongside its own half, and told apart
    # by colour: P in black (its two families being red and blue), the release
    # probability in green, ticks and axis label alike. So a tick can only be read
    # as the quantity it belongs to. The ticks are set on the parent axis across
    # the whole range (that is what draws the grid) and the negative half's labels
    # are then blanked, the green ones being drawn over them by a second axis --
    # a *labelling device* rather than a second scale (identity functions, so it
    # is the same axis relabelled, not a twin with its own units). Its spine is
    # hidden: the parent's black one spans both halves and belongs to neither.
    ticks = np.linspace(-1.0, 1.0, 11)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["" if value < 0 else f"{value:.1f}" for value in ticks])
    # Two lines each: rotated, one line of this length would run well past its own
    # half of the axis and into the other's.
    ax.set_ylabel("Protonation\n" + r"Probability, $P$", y=0.76)

    difference_axis = ax.secondary_yaxis(
        "left", functions=(lambda y: y, lambda y: y)
    )
    difference_axis.set_yticks(np.linspace(-1.0, 0.0, 6))
    difference_axis.set_ylabel(
        RELEASE_LABEL, y=0.24, color=DIFFERENCE_AXIS
    )
    difference_axis.tick_params(colors=DIFFERENCE_AXIS)
    difference_axis.spines["left"].set_visible(False)
    plt.setp(difference_axis.get_yticklabels(), fontweight="bold")

    # The two labels now stack on one side, so they are placed explicitly rather
    # than each being offset from its own tick labels: those differ in width (the
    # minus sign), which would leave the two names out of line.
    ax.yaxis.set_label_coords(YLABEL_X, 0.76, transform=ax.transAxes)
    difference_axis.yaxis.set_label_coords(YLABEL_X, 0.24, transform=ax.transAxes)

    # Names the lower half where it begins, centred under the zero line, as the
    # reaction scheme names the upper half from above it.
    ax.text(
        0.5 * (PH_MIN + PH_MAX), RELEASE_LABEL_Y, RELEASE_LABEL,
        ha="center", va="top", fontsize=RELEASE_LABEL_FONTSIZE,
        color="black", linespacing=1.35, zorder=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )

    # P = 1 above, one proton released per cycle below.
    ax.axhline(0.0, linewidth=1.0, color="black", zorder=3)
    ax.axhline(1.0, linewidth=1.0, linestyle=":", color="black", zorder=2)
    ax.axhline(-1.0, linewidth=1.0, linestyle=":", color="black", zorder=2)
    # The anchor pKa. Left unlabelled inside the panel -- both the title and the
    # legend name it, and every region of the panel is crossed by a curve.
    ax.axvline(anchor, linewidth=1.0, linestyle="--", color="black", zorder=2)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    # Reaction scheme (always drawn) on the first title line, the panel's
    # condition on the second: unlike the companion script's panels, these three
    # differ only in *how the pKas move*, so that has to be stated. Under
    # --no_title the scheme moves inside the panel and the condition is dropped.
    if not NO_TITLE:
        ax.set_title(f"{RXN_PLS}\n{condition}", fontsize=FS_TITLE, pad=8)

    # A/B/C identify a panel *within* the combined figure; a standalone figure is
    # not one of a set, so it is drawn without one (panel_label=None).
    if panel_label is not None:
        ax.text(
            -0.17, 1.12, panel_label,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_PANEL_LABEL, fontweight="bold",
        )


def plot_panel(
    ax: plt.Axes,
    panel: dict,
    entries: list[dict],
    anchor: float,
    pH: np.ndarray,
    label_panel: bool = True,
) -> None:
    """Draw one panel: the pinned titration curve, the moving ones, and delta_f.

    Solid = a state's protonation curve, red for the loaded state and blue for
    the unloaded one; dashed green = the difference the pair produces. Within
    each family the shade tracks the pKa swing. The pinned state, if any, is the
    single thick saturated curve.

    ``label_panel`` draws the A/B/C identifier; off for standalone figures.
    """
    configure_panel(
        ax, anchor,
        panel["key"] if label_panel else None,
        panel["condition"].format(anchor=anchor),
    )

    steps = np.linspace(*RAMP_SPAN, len(entries))
    loaded_shades = RAMP_LOADED(steps)
    unloaded_shades = RAMP_UNLOADED(steps)
    difference_shades = RAMP_DIFFERENCE(steps)

    # The pinned state: one curve, drawn once, thick, in its family's saturated
    # hue (red = loaded, blue = unloaded).
    if panel["fixed"] is not None:
        fixed_is_loaded = panel["fixed"] == "loaded"
        ax.plot(
            pH,
            entries[0]["f_loaded"] if fixed_is_loaded else entries[0]["f_unloaded"],
            linewidth=3.0, color=CRIMSON if fixed_is_loaded else NAVY, zorder=6,
            label=rf"{panel['fixed']} state, fixed at p$K_a$ = {anchor:g}",
        )

    # The moving state(s): thin, one curve per pKa swing, shaded by that swing.
    for entry, loaded, unloaded in zip(entries, loaded_shades, unloaded_shades):
        if panel["fixed"] != "loaded":
            ax.plot(pH, entry["f_loaded"], linewidth=1.3, color=loaded, zorder=4)
        if panel["fixed"] != "unloaded":
            ax.plot(pH, entry["f_unloaded"], linewidth=1.3, color=unloaded, zorder=4)

    # The differences, below zero, each labelled in the legend.
    for number, (entry, shade) in enumerate(zip(entries, difference_shades), start=1):
        ax.plot(
            pH, entry["delta_f"],
            linewidth=2.4, linestyle="--", color=shade, zorder=5,
            label=panel_legend_label(panel, entry, number),
        )
        # Extremum marker: black rather than the curve's own shade, which was
        # indiscernible against the green dashes it sits on.
        ax.scatter(
            [entry["ph_at_extremum"]], [entry["delta_f_min"]],
            s=32, color=EXTREMUM_COLOR, edgecolors="white", linewidths=0.8,
            zorder=8,
        )

    # Only the last (widest) swing is annotated, so the "we are at one proton"
    # result is stated without labelling every extremum. It is written on
    # whichever side of the extremum the badge column is *not* on.
    column_ph, column_values = difference_column(entries, pH)
    widest = entries[-1]
    to_the_right = widest["ph_at_extremum"] >= column_ph
    ax.annotate(
        rf"$\Delta P_{{\mathrm{{min}}}} = {widest['delta_f_min']:.2f}$",
        xy=(widest["ph_at_extremum"], widest["delta_f_min"]),
        xytext=(10 if to_the_right else -10, -8), textcoords="offset points",
        ha="left" if to_the_right else "right",
        va="top", fontsize=FS_ANNOTATION, fontweight="bold", zorder=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )

    draw_badges(ax, panel, entries, column_ph, column_values)

    if NO_TITLE:
        draw_reaction_scheme(ax)

    # Legend order: the pinned state (the first labelled artist, when there is
    # one), then the moving families, then one entry per difference curve. Placed
    # *below* the panel: with two state families plus a difference curve per pKa
    # swing, no in-panel region is left clear enough to put it inside.
    if not NO_LEGEND:
        labelled = ax.get_legend_handles_labels()[0]
        pinned = labelled[:1] if panel["fixed"] is not None else []
        ax.legend(
            handles=pinned + family_handles(panel, entries) + labelled[len(pinned):],
            loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=1,
            fontsize=FS_LEGEND, framealpha=1.0, edgecolor="0.8",
            handlelength=2.4, labelspacing=0.45, borderpad=0.7,
        )


def family_range_label(name: str, pka_values: list[float]) -> str:
    """"<name>, pKa first -> last, 0.5 steps": the interval a family walks.

    The step is stated only when it is the same between every pair of curves,
    which is the usual case but need not hold if a swing has been dropped for
    falling outside the pKa grid. Note that it is not always --delta-step: panel
    (C) splits each swing between the two states, so each moves by half of it.
    """
    span = (
        rf"{name}, p$K_a$ {pka_values[0]:g} $\rightarrow$ {pka_values[-1]:g}"
    )
    steps = {round(abs(b - a), 6) for a, b in zip(pka_values, pka_values[1:])}
    if len(steps) != 1:
        return span
    return rf"{span}, {steps.pop():g} steps"


def family_handles(panel: dict, entries: list[dict]) -> list[plt.Line2D]:
    """Legend proxies naming the thin moving families and the range they cover.

    Drawn in each family's mid shade, so the legend states which hue is the
    loaded state and which the unloaded one without relying on the ramp.
    """
    mid = float(np.mean(RAMP_SPAN))
    handles = []
    if panel["fixed"] != "loaded":
        handles.append(
            plt.Line2D(
                [], [], color=RAMP_LOADED(mid), linewidth=1.6,
                label=family_range_label(
                    "loaded states (L)", [entry["pka_loaded"] for entry in entries]
                ),
            )
        )
    if panel["fixed"] != "unloaded":
        handles.append(
            plt.Line2D(
                [], [], color=RAMP_UNLOADED(mid), linewidth=1.6,
                label=family_range_label(
                    "unloaded states (U)", [entry["pka_unloaded"] for entry in entries]
                ),
            )
        )
    return handles


def panel_legend_label(panel: dict, entry: dict, number: int) -> str:
    """Legend text for one difference curve.

    Opens with the badge number carried by all of this swing's curves, then the
    pKa swing, the pKa(s) that moved, the extremum, and the width of the uptake
    window in pH units (full width at half that extremum).
    """
    delta = rf"$\Delta$p$K_a$={entry['delta']:g}"
    peak = (
        rf"$\Delta P_{{\mathrm{{min}}}}$={entry['delta_f_min']:.2f}"
        rf", FWHM={entry['fwhm_pH']:.2f} pH"
    )
    if panel["fixed"] == "loaded":
        moved = rf"p$K_a^{{\,\mathrm{{unl}}}}$={entry['pka_unloaded']:g}"
    elif panel["fixed"] == "unloaded":
        moved = rf"p$K_a^{{\,\mathrm{{load}}}}$={entry['pka_loaded']:g}"
    else:
        moved = (
            rf"p$K_a$={entry['pka_unloaded']:g}/{entry['pka_loaded']:g}"
        )
    return f"({number})  {delta}, {moved}, {peak}"


def plot_uptake_figure(
    panel_entries: list[tuple[dict, list[dict]]],
    anchor: float,
    pH: np.ndarray,
) -> plt.Figure:
    """The three-panel uptake figure, A/B/C left to right."""
    # Extra height at the bottom for the per-panel legends under each axes.
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 9.0))

    for ax, (panel, entries) in zip(axes, panel_entries):
        plot_panel(ax, panel, entries, anchor, pH)

    # The three panels share one set of equations, so they are drawn once, above
    # the figure, rather than repeated in every panel.
    top = 1.0
    if not NO_LEGEND:
        fig.text(
            0.5, 0.995, f"{EQ_F}          {EQ_UPTAKE}          {EQ_MAX}",
            ha="center", va="top", fontsize=FS_EQUATION,
        )
        top = 0.955

    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    return fig


def plot_single_panel_figure(
    panel: dict,
    entries: list[dict],
    anchor: float,
    pH: np.ndarray,
) -> plt.Figure:
    """One panel on its own figure, for use apart from the A/B/C triptych.

    Same panel machinery, so the standalone version is the combined figure's
    panel exactly, minus its A/B/C identifier -- it is no longer one panel of a
    set. Only the equation block is re-wrapped for the narrower page.
    """
    fig, ax = plt.subplots(figsize=(7.6, 9.0))
    plot_panel(ax, panel, entries, anchor, pH, label_panel=False)

    top = 1.0
    if not NO_LEGEND:
        fig.text(
            0.5, 0.995, f"{EQ_F}          {EQ_UPTAKE}",
            ha="center", va="top", fontsize=FS_EQUATION_SINGLE,
        )
        fig.text(
            0.5, 0.958, EQ_MAX, ha="center", va="top",
            fontsize=FS_EQUATION_SINGLE,
        )
        top = 0.925

    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    return fig


def save_figure(figure: plt.Figure, output_directory: Path, stem: str) -> None:
    for suffix, kwargs in ((".png", {"dpi": DPI}), (".pdf", {})):
        figure.savefig(output_directory / f"{stem}{suffix}", bbox_inches="tight", **kwargs)


# ================================ Export ================================= #

def curves_table(
    panel_entries: list[tuple[dict, list[dict]]],
    pH: np.ndarray,
) -> pd.DataFrame:
    """Long-form table of every curve drawn in the figure."""
    frames = []
    for panel, entries in panel_entries:
        for entry in entries:
            frames.append(
                pd.DataFrame(
                    {
                        "panel": panel["key"],
                        "fixed_state": panel["fixed"] or "both_move",
                        "delta_pKa": entry["delta"],
                        "pKa_loaded": entry["pka_loaded"],
                        "pKa_unloaded": entry["pka_unloaded"],
                        "pH": pH,
                        "f_loaded": entry["f_loaded"],
                        "f_unloaded": entry["f_unloaded"],
                        "delta_f": entry["delta_f"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def maxima_table(
    panel_entries: list[tuple[dict, list[dict]]],
    anchor: float,
) -> pd.DataFrame:
    """Extremum of every difference curve: where it sits and how close to one proton."""
    rows = []
    for panel, entries in panel_entries:
        for entry in entries:
            rows.append(
                {
                    "panel": panel["key"],
                    "fixed_state": panel["fixed"] or "both_move",
                    "delta_pKa": entry["delta"],
                    "pKa_loaded": entry["pka_loaded"],
                    "pKa_unloaded": entry["pka_unloaded"],
                    "pH_at_extremum": entry["ph_at_extremum"],
                    # How far the pH of maximal uptake has been dragged from the
                    # anchor: dpKa/2 in (A) and (B), zero in (C).
                    "pH_offset_from_anchor": abs(entry["ph_at_extremum"] - anchor),
                    "delta_f_min": entry["delta_f_min"],
                    "protons_per_cycle": entry["protons_per_cycle"],
                    "delta_G_swing_kcal_per_mol": TWO_303_RT * entry["delta"],
                    # Width of the uptake window, in pH units.
                    "fwhm_pH": entry["fwhm_pH"],
                    **{
                        f"width_pH_at_dF_{level:g}": width
                        for level, width in entry["widths_pH"].items()
                    },
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the Henderson-Hasselbalch grid for every pKa 0-14 and plot "
            "proton-loading-site uptake, f_unloaded - f_loaded, for a pKa swing "
            "opened up three different ways."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # default: anchor pKa 6, swings stepped until uptake reaches ~1\n"
            "  python3 plot_pls_uptake.py\n"
            "\n"
            "  # anchor at physiological pH\n"
            "  python3 plot_pls_uptake.py --anchor 7.4\n"
            "\n"
            "  # finer ladder, stop as soon as 0.95 of a proton is taken up\n"
            "  python3 plot_pls_uptake.py --delta-step 0.5 --uptake-target 0.95\n"
            "\n"
            "  # narrowed x-axis pH range (the CSVs still span pH 0-14)\n"
            "  python3 plot_pls_uptake.py --ph-min 2 --ph-max 12\n"
            "\n"
            "  # start the pKa ladder somewhere else, over a matching pH range\n"
            "  python3 plot_pls_uptake.py --anchor 4 --ph-min 0 --ph-max 10\n"
            "\n"
            "  # bare panels: no equation block, no curve legends\n"
            "  python3 plot_pls_uptake.py --no_legend\n"
            "\n"
            "  # figure for a manuscript: everything but the curves in the caption\n"
            "  python3 plot_pls_uptake.py --no_legend --no_title\n"
        ),
    )
    parser.add_argument(
        "--anchor", type=float, default=PKA_ANCHOR,
        help="Anchor pKa the ladder starts from: the fixed state in (A)/(B), the "
             "centre of symmetry in (C) (default: %(default)s).",
    )
    parser.add_argument(
        "--delta-step", type=float, default=DELTA_STEP,
        help="Increment of the pKa swing between successive curves "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--uptake-target", type=float, default=UPTAKE_TARGET,
        help="Stop adding curves once |delta_f| reaches this many protons "
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
        "--outdir", type=Path, default=Path("plot_pls_uptake"),
        help="Directory for every generated file (default: %(default)s).",
    )
    parser.add_argument(
        "--no_legend", action="store_true",
        help="Omit the equation block above the figure and the curve legend below "
             "each panel. Titles, reaction schemes and badges are kept.",
    )
    parser.add_argument(
        "--no_title", action="store_true",
        help="Drop the panel titles; the reaction scheme they carry is then drawn "
             "inside the panel instead (the line naming which pKa moves is lost, "
             "and belongs in the figure caption).",
    )
    return parser.parse_args()


def main() -> None:
    global PH_MIN, PH_MAX, NO_LEGEND, NO_TITLE

    args = parse_args()
    if args.ph_min >= args.ph_max:
        raise SystemExit(
            f"--ph-min ({args.ph_min}) must be less than --ph-max ({args.ph_max})."
        )
    if not 0.0 < args.uptake_target < 1.0:
        raise SystemExit("--uptake-target must lie strictly between 0 and 1.")
    if args.delta_step <= 0.0:
        raise SystemExit("--delta-step must be positive.")
    if not PKA_GRID_MIN <= args.anchor <= PKA_GRID_MAX:
        raise SystemExit(
            f"--anchor ({args.anchor}) must lie within pKa "
            f"{PKA_GRID_MIN:g}-{PKA_GRID_MAX:g}."
        )
    PH_MIN, PH_MAX = args.ph_min, args.ph_max
    NO_LEGEND = args.no_legend
    NO_TITLE = args.no_title

    output_directory = args.outdir
    output_directory.mkdir(parents=True, exist_ok=True)

    # The Henderson-Hasselbalch grid always spans the full pKa 0-14 x pH 0-14
    # range, independent of the plot window.
    grid = henderson_hasselbalch_grid()
    grid.to_csv(output_directory / "henderson_hasselbalch.csv", index=False)

    pH = np.linspace(PH_MIN, PH_MAX, N_POINTS)
    deltas = delta_ladder(args.uptake_target, args.delta_step)
    panel_entries = [
        (panel, panel_curves(panel, deltas, args.anchor, pH)) for panel in PANELS
    ]
    empty = [panel["key"] for panel, entries in panel_entries if not entries]
    if empty:
        raise SystemExit(
            f"anchor pKa {args.anchor:g} leaves panel(s) {', '.join(empty)} with no "
            f"pKa pair inside {PKA_GRID_MIN:g}-{PKA_GRID_MAX:g}; move --anchor "
            f"towards the middle of the range."
        )

    curves_table(panel_entries, pH).to_csv(
        output_directory / "pls_uptake_curves.csv", index=False
    )
    maxima = maxima_table(panel_entries, args.anchor)
    maxima.to_csv(output_directory / "pls_uptake_maxima.csv", index=False)

    save_figure(
        plot_uptake_figure(panel_entries, args.anchor, pH),
        output_directory, "figure_pls_uptake",
    )
    # ... and each panel again on its own, for use outside the triptych.
    for panel, entries in panel_entries:
        save_figure(
            plot_single_panel_figure(panel, entries, args.anchor, pH),
            output_directory, f"figure_pls_uptake_{panel['key']}",
        )

    print(f"pKa swings plotted: {', '.join(f'{d:g}' for d in deltas)}")
    print(maxima.to_string(index=False))
    print(f"written to {output_directory}/")

    plt.show()


if __name__ == "__main__":
    main()
