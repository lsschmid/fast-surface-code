"""Overview figure: a single rotated surface-code patch with its parts labelled.

Draws one rectangular patch (dx x dz) in the same style and palette as the
error-propagation figure: data qubits, X-/Z-stabilizer surfaces (semicircles on
the boundary), the logical operators X_L / Z_L as arrows, and the dx / dz
directions. Companion to Sec. II-A of the paper.

Output: a single-column vector PDF. Run:
    uv run python3 evaluation/plot_patch_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.surface_code import SurfaceCode

# Same patch shape and orientation as the construction figure (Fig. 1):
# dx rows are drawn horizontally (TRANSPOSE), dz columns vertically.
DX, DZ = 5, 3
OUT = Path(__file__).parent / "patch_overview.pdf"

# Palette matched to the detector-slice / error-propagation figures.
C_IDLE = "#7a7a7a"
C_X = "#c1272d"
C_Z = "#1f5fbf"
C_XSTAB = "#e8a0a0"
C_ZSTAB = "#a8c4ec"
STAB_ALPHA = 0.45


def pt(rc):
    """(row, col) -> (x, y), transposed to match the detslice orientation."""
    r, c = rc
    return (r, c)


def draw_stab(ax, mq, cells, color):
    """One stabilizer surface: polygon in the bulk, semicircle on the boundary."""
    pts = [np.array(pt(rc), float) for rc in cells]
    if len(pts) >= 3:
        ctr = sum(pts) / len(pts)
        pts.sort(key=lambda p: np.arctan2(p[1] - ctr[1], p[0] - ctr[0]))
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                             alpha=STAB_ALPHA, zorder=0))
    else:
        p0, p1 = pts
        mqxy = np.array(pt(mq), float)
        ctr = (p0 + p1) / 2
        r = np.linalg.norm(p1 - p0) / 2
        d = (p1 - p0) / (2 * r)
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, mqxy - ctr) < 0:
            nrm = -nrm
        arc = [ctr + r * (np.cos(t) * d + np.sin(t) * nrm)
               for t in np.linspace(0, np.pi, 20)]
        ax.add_patch(Polygon(arc, closed=True, facecolor=color, edgecolor="none",
                             alpha=STAB_ALPHA, zorder=0))


def main() -> None:
    code = SurfaceCode(dx=DX, dz=DZ)
    data = list(code.qubits)
    coord = {i: code.qubit_coords[q] for i, q in enumerate(data)}

    def stab_list(stab):
        return [(code.qubit_coords[mq], [code.qubit_coords[dq] for dq in ds])
                for mq, ds in stab.items()]

    ex, ey = DX, DZ  # drawing extents (rows horizontal, cols vertical)
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.set_aspect("equal")
    ax.axis("off")

    for mq, cells in stab_list(code.stabilizers):
        draw_stab(ax, mq, cells, C_XSTAB)
    for mq, cells in stab_list(code.z_stabilizers):
        draw_stab(ax, mq, cells, C_ZSTAB)

    # stabilizer-type labels on one bulk plaquette of each type
    for stabs, color, lbl in ((stab_list(code.stabilizers), C_X, "$S_X$"),
                              (stab_list(code.z_stabilizers), C_Z, "$S_Z$")):
        mq, cells = max(stabs, key=lambda s: len(s[1]))
        ctr = sum(np.array(pt(rc), float) for rc in cells) / len(cells)
        ax.text(*ctr, lbl, color=color, fontsize=11, ha="center", va="center",
                weight="bold")

    for i in range(len(data)):
        x, y = pt(coord[i])
        ax.scatter([x], [y], s=55, color=C_IDLE, zorder=1)

    # logicals as arrows beside the patch (same placement as the error figure):
    # Z_L vertical along a column (weight dz), X_L horizontal along a row (dx).
    ax.annotate("", xy=(-1.0, ey - 1), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_Z, lw=2.0))
    ax.text(-1.35, (ey - 1) / 2, "$Z_L$", color=C_Z, rotation=90,
            ha="right", va="center", fontsize=12)
    ax.annotate("", xy=(ex - 1, -1.0), xytext=(0, -1.0),
                arrowprops=dict(arrowstyle="-|>", color=C_X, lw=2.0))
    ax.text((ex - 1) / 2, -1.35, "$X_L$", color=C_X, ha="center",
            va="bottom", fontsize=12)

    # distance annotations on the opposite sides, clear of the boundary bulges
    ax.annotate("", xy=(ex - 1, ey - 0.15), xytext=(0, ey - 0.15),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text((ex - 1) / 2, ey + 0.15, f"$d_x = {DX}$", ha="center", va="top",
            fontsize=10)
    ax.annotate("", xy=(ex - 0.15, ey - 1), xytext=(ex - 0.15, 0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(ex + 0.15, (ey - 1) / 2, f"$d_z = {DZ}$", rotation=270,
            ha="left", va="center", fontsize=10)

    ax.set_xlim(-1.6, ex + 0.55)
    ax.set_ylim(ey + 0.5, -1.5)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
