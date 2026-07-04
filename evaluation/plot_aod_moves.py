"""AOD atom-move figure for the unitary preparation cascade.

Draws the preparation as a sequence of neutral-atom moves: one panel per circuit
layer, showing the data-qubit array, which atoms shift (a dashed ghost at the
vacated site + an arrow to the new position), and the CZ bonds formed once the
moving atom nearly abuts its partner.

Move model (in-place nearest-neighbour shift, AOD-friendly, pickup-minimizing):
  * Pairing layer (two atoms in the same line): the *left* atom of each pair
    hops right to meet its partner, so every mover goes the same direction --
    a single collective AOD shift.
  * Cascade layer (two atoms in adjacent lines): the *control* atom (the moving
    front, e.g. the middle seed line) hops one lattice step toward its target.
    The middle line therefore moves down and then up on consecutive steps, i.e.
    it is picked up once and serves both neighbours -- minimizing pickups/drops.

A global CZ then entangles all adjacent pairs at once; atoms return afterwards.

Output is a vector PDF (matplotlib). Run:
    uv run python3 evaluation/plot_aod_moves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.surface_code import SurfaceCode

# ======================================================================
# Parameters — edit these
# ======================================================================
DX = 5                       # X-distance = #rows   (odd, >= 3)
DZ = 3                       # Z-distance = #cols   (odd, >= 3)
PLUS = True                  # |0>_L if False, |+>_L if True
METHOD = "bidirectional"     # "rows" | "bidirectional" | "sequential"
SHOW_INITIAL = True          # include an "initial array" panel
GRID_LINES = True            # dashed AOD row/column guide lines
TOUCH_GAP = 0.26             # final gap between two interacting atoms (lattice units)
PANEL_IN = 1.7               # panel size (inches)
N_ROWS = 2                   # panel-grid rows (1 = single strip); an empty
                             # grid cell, if any, holds the legend
OUT_PATH = Path(__file__).parent / "aod_moves.pdf"

# Colors
C_IDLE = "#d9d9d9"           # atoms not involved in this layer
C_FIXED = "#264653"          # stationary atom of an interacting pair
C_MOVED = "#e76f51"          # atom that hops
C_BOND = "#2a9d8f"           # CZ bond
C_GRID = "#bfbfbf"           # AOD guide lines
# ======================================================================


def cx_layers(circ) -> list[list[tuple[int, int]]]:
    """Extract the CX layers as lists of (control, target) qubit pairs."""
    layers = []
    for inst in circ:
        if inst.name == "CX":
            t = [x.value for x in inst.targets_copy()]
            layers.append([(t[i], t[i + 1]) for i in range(0, len(t), 2)])
    return layers


def mover_and_anchor(ctrl, tgt, coords, lines_are_rows, mid_line):
    """Which atom hops (mover) and which stays (anchor), per the move model.

    ``lines_are_rows`` says whether the cascade sweeps rows (|+>) or columns
    (|0>). A pairing gate acts *within* a line; a cascade gate acts *between*
    adjacent lines.

    Pairing: the lower-index atom hops toward its partner (one uniform
    direction -- a single collective AOD shift).

    Cascade: if the pair touches the middle seed line, the middle atom hops (it
    is picked up once and serves both of its neighbours); otherwise the atom
    farther from the centre hops *inward* toward the established block. This
    keeps the built block stationary and shuttles fresh outer lines in, which
    generalizes to larger patches. ``mid_line`` is the seed line index, or None
    (non-bidirectional) to fall back to "control moves".
    """
    cr, cc = coords[ctrl]
    tr, tc = coords[tgt]
    is_pairing = (cr == tr) if lines_are_rows else (cc == tc)
    if is_pairing:
        if cr == tr:  # within-row pair: left atom moves right
            return (ctrl, tgt) if cc < tc else (tgt, ctrl)
        return (ctrl, tgt) if cr < tr else (tgt, ctrl)  # within-col: top moves down

    # cascade
    line_c = cr if lines_are_rows else cc
    line_t = tr if lines_are_rows else tc
    if mid_line is None:
        return ctrl, tgt  # fallback: control moves toward target
    if line_c == mid_line:
        return ctrl, tgt  # middle serves this neighbour
    if line_t == mid_line:
        return tgt, ctrl
    # neither on the seed line: the farther atom moves inward
    if abs(line_c - mid_line) > abs(line_t - mid_line):
        return ctrl, tgt
    return tgt, ctrl


def main() -> None:
    code = SurfaceCode(dx=DX, dz=DZ, plus=PLUS)
    getattr(code, f"unitary_init_{METHOD}")()
    circ = code._circuit

    coords = code.qubit_coords            # qubit -> (row, col)
    data = list(code.qubits)
    layers = cx_layers(circ)

    # A cascade layer couples adjacent *lines*; a pairing layer couples within a
    # line. Detect per layer whether control moves (cascade) or the left/upper
    # atom moves (pairing) using the mover_and_anchor rule directly.
    def xy(q):
        r, c = coords[q]
        return c, r

    rows = sorted({coords[q][0] for q in data})
    cols = sorted({coords[q][1] for q in data})
    home = {q: xy(q) for q in data}

    # Seed (middle) line index for the bidirectional cascade; None otherwise.
    n_lines = DX if PLUS else DZ
    mid_line = (n_lines - 1) // 2 if METHOD == "bidirectional" else None
    # Atoms that moved in the previous layer keep their displaced position, so a
    # multi-step atom (e.g. the middle line: down then up) moves continuously.
    last_end: dict[int, tuple[float, float]] = {}

    n_panels = len(layers) + (1 if SHOW_INITIAL else 0)
    n_cols = -(-n_panels // N_ROWS)  # ceil division
    panel_h = PANEL_IN * len(rows) / max(len(cols), 1) * 0.9
    fig, axgrid = plt.subplots(
        N_ROWS, n_cols,
        figsize=(PANEL_IN * n_cols, panel_h * N_ROWS), squeeze=False,
    )
    axes = [ax for row in axgrid for ax in row]
    spare = axes[n_panels:]   # unused grid cells (legend goes into the first)
    axes = axes[:n_panels]

    titles, panels = [], []
    if SHOW_INITIAL:
        titles.append("initial array")
        panels.append(None)
    for i in range(len(layers)):
        titles.append(f"layer {i + 1}")
        panels.append(layers[i])

    for ax, title, layer in zip(axes, titles, panels):
        ax.set_title(title, fontsize=16)
        ax.set_aspect("equal")
        ax.axis("off")

        if GRID_LINES:
            for r in rows:
                ax.plot([min(cols), max(cols)], [r, r], color=C_GRID, lw=0.8,
                        ls=(0, (4, 3)), zorder=0)
            for c in cols:
                ax.plot([c, c], [min(rows), max(rows)], color=C_GRID, lw=0.8,
                        ls=(0, (4, 3)), zorder=0)

        moved = set()
        if layer is None:
            last_end = {}
        else:
            new_end: dict[int, tuple[float, float]] = {}
            for ctrl, tgt in layer:
                mover, anchor = mover_and_anchor(ctrl, tgt, coords, PLUS, mid_line)
                ax0, ay0 = home[anchor]
                # mover starts where it ended last layer (continuous trajectory)
                ox, oy = last_end.get(mover, home[mover])
                # mover stops TOUCH_GAP short of the anchor, on its approach side
                dx, dy = ox - ax0, oy - ay0
                dist = (dx * dx + dy * dy) ** 0.5 or 1.0
                mxg = ax0 + TOUCH_GAP * dx / dist
                myg = ay0 + TOUCH_GAP * dy / dist
                # CZ bond between anchor and the shifted mover
                ax.plot([ax0, mxg], [ay0, myg], color=C_BOND, lw=3.0, zorder=1)
                # dashed ghost at the vacated site + arrow to the new position
                ax.scatter([ox], [oy], s=130, facecolors="none", edgecolors=C_MOVED,
                           linewidths=1.4, alpha=0.45, linestyle="--", zorder=2)
                ax.annotate("", xy=(mxg, myg), xytext=(ox, oy),
                            arrowprops=dict(arrowstyle="-|>", color=C_MOVED, lw=2.0),
                            zorder=2)
                ax.scatter([mxg], [myg], s=150, color=C_MOVED, zorder=4)  # moved
                ax.scatter([ax0], [ay0], s=150, color=C_FIXED, zorder=4)  # stationary
                moved.update([mover, anchor])
                new_end[mover] = (mxg, myg)
            last_end = new_end

        idle = [q for q in data if q not in moved]
        if idle:
            xs, ys = zip(*(home[q] for q in idle))
            ax.scatter(xs, ys, s=120, color=C_IDLE, zorder=3)

        ax.invert_yaxis()
        ax.margins(0.18)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FIXED,
               markersize=12, label="stationary"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MOVED,
               markersize=12, label="moved"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=C_MOVED, markersize=12, label="vacated site"),
        Line2D([0], [0], color=C_BOND, lw=2.2, label="CZ"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IDLE,
               markersize=12, label="idle"),
    ]
    if spare:
        for ax in spare:
            ax.axis("off")
        spare[0].legend(handles=handles, loc="center", ncol=1, frameon=False,
                        fontsize=14)
        fig.tight_layout()
    else:
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({n_panels} panels)")


if __name__ == "__main__":
    main()
