"""Plot logical error rate vs noise strength from evaluation results.

The x-axis adapts to the noise model recorded in the results: the physical error
rate ``p`` for the depolarizing model, or the noise scaling factor for the
gemini (neutral-atom hardware) model.

Encoding: each construction (method) has its own colour *and* line style;
the code distance (patch shape) is shown by line opacity. No markers, so the
same-method curves are easy to compare as a group.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

# Compact fonts on a figure sized to the two-column width, so text stays
# legible at \textwidth without \includegraphics down-scaling it.
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

METHOD_COLORS = {
    "rows": "tab:blue",
    "bidirectional": "tab:orange",
    "sequential": "tab:red",
    "measurement": "tab:green",
}
METHOD_LS = {
    "bidirectional": "-",
    "rows": "--",
    "sequential": "-.",
    "measurement": ":",
}
# legend labels matching the paper's terminology
METHOD_NAMES = {
    "rows": "unidirectional",
    "sequential": "single-pivot",
}
GEMINI_YLIM = (1e-5, 1.5)


def noise_model_of(results: list[dict]) -> str:
    """Noise model recorded in the results (older results imply depolarizing)."""
    return results[0].get("noise_model", "depolarizing") if results else "depolarizing"


def shape_of(r: dict) -> str:
    """Patch-shape label, tolerant of old results that only stored ``d``."""
    return r.get("shape") or f"{r['d']}x{r['d']}"


def shape_key(shape: str) -> tuple[int, int]:
    """Numeric sort key for a 'dxxdz' label (so 3x3 < 7x7 < 11x11)."""
    return tuple(int(v) for v in shape.split("x"))


def dist_alpha(shape: str, all_shapes: list[str]) -> float:
    """Line opacity for a distance: faint (small d) -> opaque (large d)."""
    if len(all_shapes) <= 1:
        return 1.0
    lo, hi = 0.45, 1.0
    i = all_shapes.index(shape)
    return lo + (hi - lo) * i / (len(all_shapes) - 1)


def plot_basis(
    ax: plt.Axes,
    results: list[dict],
    basis: str,
    title: str,
    all_shapes: list[str],
    noise_model: str = "depolarizing",
) -> None:
    """Plot one basis (zero or plus) on the given axes."""
    pts = [r for r in results if r.get("basis", "zero") == basis]
    methods = sorted({r["method"] for r in pts})
    shapes = sorted({shape_of(r) for r in pts}, key=shape_key)

    for method in methods:
        for shape in shapes:
            data = sorted(
                (r for r in pts if r["method"] == method and shape_of(r) == shape),
                key=lambda r: r["p"],
            )
            xy = [(r["p"], r["rate"]) for r in data if r["rate"] > 0]
            if not xy:
                continue
            xs, ys = zip(*xy)
            ax.plot(
                xs, ys,
                color=METHOD_COLORS.get(method, "tab:gray"),
                linestyle=METHOD_LS.get(method, "-"),
                alpha=dist_alpha(shape, all_shapes),
                linewidth=1.8,
                solid_capstyle="round",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    if noise_model == "gemini":
        xs = sorted({r["p"] for r in pts})
        if xs:
            ax.set_xlim(min(xs) / 1.5, max(xs) * 1.5)
            # label every measured scaling factor with a plain number
            # (log axis alone would only show 10^0)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylim(*GEMINI_YLIM)
        # x-axis meaning is stated in the LaTeX caption -> no in-plot label
    else:
        # power-law reference lines only make sense against a physical rate p
        p_range = np.logspace(-4, -2, 50)
        ax.plot(p_range, p_range, "k--", alpha=0.3, linewidth=0.8)
        ax.plot(p_range, 10 * p_range**2, "k--", alpha=0.2, linewidth=0.8)
        ax.plot(p_range, 100 * p_range**3, "k--", alpha=0.15, linewidth=0.8)
        ax.set_xlim(1e-4, 1e-2)
        ax.set_ylim(1e-9, 1e-2)
        ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate $p_L$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def main() -> None:
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # |0> panel on the left, |+> on the right (alphabetical would flip them)
    bases = sorted({r.get("basis", "zero") for r in results},
                   key=lambda b: b != "zero")
    methods = sorted({r["method"] for r in results})
    shapes = sorted({shape_of(r) for r in results}, key=shape_key)
    noise_model = noise_model_of(results)

    if len(bases) == 1:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        label = "|0⟩" if bases[0] == "zero" else "|+⟩"
        plot_basis(ax, results, bases[0], f"{label} state", shapes, noise_model)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), sharey=True)
        for ax, basis in zip(axes, bases):
            label = "|0⟩" if basis == "zero" else "|+⟩"
            plot_basis(ax, results, basis, f"{label} state", shapes, noise_model)
        axes[1].set_ylabel("")
        if noise_model != "gemini":  # single shared x-label (gemini: see caption)
            for ax in axes:
                ax.set_xlabel("")
            fig.supxlabel("Physical error rate $p$", fontsize=8.5)

    # Two legends: construction (colour + line style), distance (opacity).
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linestyle=METHOD_LS[m],
               linewidth=2, label=METHOD_NAMES.get(m, m))
        for m in methods if m in METHOD_COLORS
    ]
    shape_handles = [
        Line2D([0], [0], color="gray", linewidth=2.2, alpha=dist_alpha(s, shapes),
               label=s)
        for s in shapes
    ]
    leg1 = axes[0].legend(handles=method_handles, loc="upper left", title="Method")
    axes[0].add_artist(leg1)
    axes[0].legend(handles=shape_handles, loc="lower right", title="Distance (dx×dz)")

    fig.tight_layout()
    out_path = Path(__file__).parent / "plot.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
