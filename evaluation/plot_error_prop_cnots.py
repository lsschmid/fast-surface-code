"""Error propagation *through the CNOT cascade*, one strip figure per case.

For a single-qubit fault introduced before the cascade, this draws a strip of
panels (initial + one per CNOT layer) in the same style as the stabilizer /
AOD-move figures: each panel shows that layer's CNOTs (control -> target), with
the gates that actually propagate the error highlighted, and the error support
after the layer. Watching the strip shows the error spreading along the CNOTs.

Propagation rule (Heisenberg): through CNOT(c, t), X on the control copies to
the target (X_c -> X_c X_t) and Z on the target copies to the control
(Z_t -> Z_c Z_t); a copy onto a qubit that already carries the error cancels.

The final support is classified (reduced modulo the stabilizers, and modulo the
logical Z_L for Z on |0>) into: stabilizer / equivalent-to-weight-w / logical.

Output: one vector PDF per case. Run:
    uv run python3 evaluation/plot_error_prop_cnots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.surface_code import SurfaceCode

# ======================================================================
# Parameters
# ======================================================================
D = 5
METHOD = "bidirectional"
# Transpose the drawing (row <-> col) so the |0> column-build is displayed as a
# row-build, matching the orientation of the detector-slice figures.
TRANSPOSE = True
# One figure per error type; each shows two labelled faults (a), (b) in the same
# patch. Entry: (label, error type, origin cell (row, col)).
FIGURES = {
    "X": [                              # protected direction
        ("a", "X", (1, 0)),             # spreads into a stabilizer (no error)
        ("b", "X", (4, 4)),             # weight-2, correctable
    ],
    "Z": [                              # conjugate direction
        ("a", "Z", (0, 2)),             # spreads to the full logical (no error)
        ("b", "Z", (4, 3)),             # weight-2 -> NOT fault tolerant
    ],
}
OUT_DIR = Path(__file__).parent
PANEL_IN = 1.7

# Palette matched to the stim detector-slice figures: X-type is red, Z-type blue.
C_IDLE = "#c9c9c9"          # data qubit
C_CNOT = "#1a1a1a"          # inactive CNOT (near-black for visibility)
C_X = "#c1272d"             # X error + propagating CNOT (red, X-type)
C_Z = "#1f5fbf"             # Z error + propagating CNOT (blue, Z-type)
C_XSTAB = "#e8a0a0"         # X-stabilizer surface (light red)
C_ZSTAB = "#a8c4ec"         # Z-stabilizer surface (light blue)
STAB_ALPHA = 0.45           # background stabilizer surfaces
# ======================================================================


def pt(rc):
    """(row, col) -> (x, y), transposed to match the detslice orientation."""
    r, c = rc
    return (r, c) if TRANSPOSE else (c, r)


def draw_stab(ax, mq, cells, color):
    """Shade one stabilizer as a faint surface: a polygon for bulk plaquettes,
    an outward-bulging semicircle for weight-2 boundary plaquettes. ``mq`` (the
    measure-qubit site) sets the outward direction of the semicircle."""
    pts = [np.array(pt(rc), float) for rc in cells]
    if len(pts) >= 3:
        ctr = sum(pts) / len(pts)
        pts.sort(key=lambda p: np.arctan2(p[1] - ctr[1], p[0] - ctr[0]))
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                             alpha=STAB_ALPHA, zorder=0))
    elif len(pts) == 2:
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


def cx_layers(circ, pos):
    layers = []
    for inst in circ:
        if inst.name == "CX":
            t = [x.value for x in inst.targets_copy()]
            layers.append([(pos[t[i]], pos[t[i + 1]]) for i in range(0, len(t), 2)])
    return layers


def classify(support, etype, Xst, Zst, ZL):
    def min_weight(v, gens, extra=None):
        G = list(gens) + ([extra] if extra is not None else [])
        best = int(v.sum())
        for m in range(1 << len(G)):
            w = v.copy()
            for b in range(len(G)):
                if m >> b & 1:
                    w = w ^ G[b]
            best = min(best, int(w.sum()))
        return best
    if etype == "X":
        mw = min_weight(support, Xst)
        return "stabilizer (no error)" if mw == 0 else f"$\\equiv$ weight {mw} (correctable)"
    mw = min_weight(support, Zst, extra=ZL)
    if mw == 0:
        return "$\\equiv \\mathbb{1}$ (no error)"
    if mw == 1:
        return "$\\equiv$ weight 1 (fault tolerant)"
    return f"weight {mw} (NOT fault tolerant)"


def build_figure(name, errors, code, layers, coord, cell_to_i, meta):
    """Draw one figure showing several same-type faults propagating together."""
    n = len(coord)
    etype = errors[0][1]
    color = C_X if etype == "X" else C_Z

    # propagate every error independently; record per-layer support + active CNOTs
    supps = [np.zeros(n, dtype=int) for _ in errors]
    for k, (_, _, origin) in enumerate(errors):
        supps[k][cell_to_i[origin]] = 1
    states = [([s.copy() for s in supps], [[] for _ in errors])]
    for layer in layers:
        active = [[] for _ in errors]
        for k in range(len(errors)):
            pre = supps[k].copy()
            for c, t in layer:
                if etype == "X" and pre[c]:
                    supps[k][t] ^= 1
                    active[k].append((c, t))
                elif etype == "Z" and pre[t]:
                    supps[k][c] ^= 1
                    active[k].append((c, t))
        states.append(([s.copy() for s in supps], active))

    verdicts = [classify(supps[k], etype, meta["Xst"], meta["Zst"], meta["ZL"])
                for k in range(len(errors))]

    nr = len({c[0] for c in coord.values()})
    nc = len({c[1] for c in coord.values()})
    ex = nr if TRANSPOSE else nc
    ey = nc if TRANSPOSE else nr
    xlim = (-2.4, ex - 0.5)
    # axes-fraction x of the patch centre, so titles sit over the patch (not the
    # axes, which are widened on the left for the Z_L label)
    title_x = ((ex - 1) / 2 - xlim[0]) / (xlim[1] - xlim[0])
    n_panels = len(states)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(PANEL_IN * n_panels, PANEL_IN * ey / ex + 0.8))
    titles = ["initial"] + [f"layer {i + 1}" for i in range(len(layers))]

    for idx, (ax, (supp_list, active)) in enumerate(zip(axes, states)):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[idx], fontsize=11, x=title_x)

        for mq, cells in meta["x_stabs"]:
            draw_stab(ax, mq, cells, C_XSTAB)
        for mq, cells in meta["z_stabs"]:
            draw_stab(ax, mq, cells, C_ZSTAB)

        # logicals: drawn once beside the first patch, in their colors.
        # Z_L vertical on the left; X_L horizontal on top with its label at the
        # right end so it clears the panel title.
        if idx == 0:
            ax.annotate("", xy=(-1.1, ey - 1), xytext=(-1.1, 0),
                        arrowprops=dict(arrowstyle="-|>", color=C_Z, lw=2.0))
            ax.text(-1.45, (ey - 1) / 2, "$Z_L$", color=C_Z, rotation=90,
                    ha="right", va="center", fontsize=12)
            ax.annotate("", xy=(ex - 1, -1.1), xytext=(0, -1.1),
                        arrowprops=dict(arrowstyle="-|>", color=C_X, lw=2.0))
            ax.text(ex - 1, -1.45, "$X_L$", color=C_X, ha="center",
                    va="bottom", fontsize=12)

        # CNOTs (active if they propagate any error this layer)
        if idx > 0:
            active_union = {cnot for a in active for cnot in a}
            for c, t in layers[idx - 1]:
                (cx, cy), (tx, ty) = pt(coord[c]), pt(coord[t])
                on = (c, t) in active_union
                ax.plot([cx, tx], [cy, ty], color=(color if on else C_CNOT),
                        lw=(3.2 if on else 1.8), zorder=(3 if on else 2))
                ax.scatter([cx], [cy], s=28, color=(color if on else C_CNOT),
                           zorder=3)
                ax.scatter([tx], [ty], s=60, facecolors="white",
                           edgecolors=(color if on else C_CNOT),
                           linewidths=1.6, zorder=3)

        # lattice
        for i in range(n):
            x, y = pt(coord[i])
            ax.scatter([x], [y], s=55, color=C_IDLE, zorder=1)

        # each error's support; (a)/(b) label only on the initial panel, nudged
        # outward (away from patch centre) with a white halo so it sits in the
        # margin and does not cover the lattice.
        cx, cy = (ex - 1) / 2, (ey - 1) / 2
        for k, (lbl, _, origin) in enumerate(errors):
            pts = [pt(coord[i]) for i in range(n) if supp_list[k][i]]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=150,
                           color=color, edgecolors="white", linewidths=1.0,
                           zorder=4)
            if idx == 0:
                ox, oy = pt(origin)
                # place the label diagonally toward the patch centre, in the gap
                # between lattice sites so it clears both the error circle and
                # the grey data dots
                sx = 0.85 if ox <= cx else -0.85
                sy = 0.85 if oy <= cy else -0.85
                t = ax.text(ox + sx, oy + sy, f"({lbl})", fontsize=11,
                            weight="bold", color=color, ha="center",
                            va="center", zorder=6)
                t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

        ax.set_xlim(*xlim)
        ax.set_ylim(ey - 0.5, -1.9)

    caption = ";  ".join(f"({lbl}) {v}" for (lbl, _, _), v in zip(errors, verdicts))
    fig.tight_layout()
    out = OUT_DIR / f"error_prop_{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}: {caption}")


def main() -> None:
    code = SurfaceCode(d=D, plus=False)
    getattr(code, f"unitary_init_{METHOD}")()
    data = list(code.qubits)
    n = len(data)
    pos = {q: i for i, q in enumerate(data)}
    coord = {i: code.qubit_coords[data[i]] for i in range(n)}
    cell_to_i = {code.qubit_coords[data[i]]: i for i in range(n)}
    layers = cx_layers(code._circuit, pos)

    def stab_vecs(stab):
        out = []
        for _, ds in stab.items():
            w = np.zeros(n, dtype=int)
            for dq in ds:
                w[pos[dq]] = 1
            out.append(w)
        return out

    def stab_list(stab):
        return [(code.qubit_coords[mq], [code.qubit_coords[dq] for dq in ds])
                for mq, ds in stab.items()]

    ZL = np.array(code.get_observable(), dtype=int)
    XL = np.array(SurfaceCode(d=D, plus=True).get_observable(), dtype=int)
    meta = {
        "Xst": stab_vecs(code.stabilizers),
        "Zst": stab_vecs(code.z_stabilizers),
        "ZL": ZL,
        "x_stabs": stab_list(code.stabilizers),
        "z_stabs": stab_list(code.z_stabilizers),
    }

    for name, errors in FIGURES.items():
        build_figure(name, errors, code, layers, coord, cell_to_i, meta)


if __name__ == "__main__":
    main()
