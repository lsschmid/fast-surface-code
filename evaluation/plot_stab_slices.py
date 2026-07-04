"""Stabilizer-construction figure: stim detector-slices packed into rows.

Renders the preparation cascade as a grid of stim ``detslice-with-ops`` panels
(one column per circuit tick), with the Z-type stabilizers in one row and the
X-type stabilizers in another. The Z / X panels come from reading the *same*
preparation out in the Z / X basis (a single terminal readout only yields one
detector type, so each stabilizer family needs its own readout).

The panels are stim SVGs, composed into one parent SVG via nested ``<svg>``
elements, so the result stays fully vector (good for a paper). Every panel is
cropped to a common content bounding box (computed across all panels, so the
lattice stays aligned) to remove stim's generous margins. Output is SVG;
convert to PDF with e.g. ``rsvg-convert``/Inkscape if you need it.

Run:  uv run python3 evaluation/plot_stab_slices.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.surface_code import SurfaceCode

# ======================================================================
# Parameters — edit these
# ======================================================================
DX = 5                       # X-distance = #rows   (odd, >= 3)
DZ = 3                       # Z-distance = #cols   (odd, >= 3)
PLUS = True                  # |0>_L if False, |+>_L if True
METHOD = "bidirectional"     # "rows" | "bidirectional" | "sequential"
BASES = ("Z", "X")           # stabilizer rows to draw, top to bottom
INCLUDE_READOUT = True       # include the final measurement column
TITLE = ""                   # "" -> no title; None -> auto; or a custom string
OUT_PATH = Path(__file__).parent / "stab_slices.svg"

# Layout (SVG user units)
CELL_H = 200.0               # panel height; width follows the cropped aspect ratio
COL_GAP = 6.0
ROW_GAP = 10.0
LEFT = 48.0                  # room for row labels
PAD = 8.0                    # outer padding
CROP_PAD = 12.0              # padding kept around the panel content
# ======================================================================


def build_circuit(x_basis: bool) -> stim.Circuit:
    """Preparation + final measurement, read out in the given basis."""
    code = SurfaceCode(dx=DX, dz=DZ, plus=PLUS)
    getattr(code, f"unitary_init_{METHOD}")()
    code.final_measurement(x_basis=x_basis)
    return code._circuit


def panel_inner(circ: stim.Circuit, tick: int) -> str:
    """Inner SVG markup for one detslice tick (stim's 'Tick N' label removed)."""
    svg = str(circ.diagram("detslice-with-ops-svg", tick=range(tick, tick + 1)))
    m = re.match(r'<svg[^>]*>(.*)</svg>\s*$', svg, re.DOTALL)
    if m is None:
        raise ValueError("Unexpected stim SVG format")
    # Drop stim's per-panel "Tick N" caption; we add our own column captions.
    return re.sub(r"<text[^>]*>Tick\s*\d+</text>", "", m.group(1))


def content_bbox(inner: str) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of the drawn content in an inner SVG."""
    xs = [float(v) for v in re.findall(r'cx="(-?[\d.]+)"', inner)]
    ys = [float(v) for v in re.findall(r'cy="(-?[\d.]+)"', inner)]
    xs += [float(v) for v in re.findall(r'\bx="(-?[\d.]+)"', inner)]
    ys += [float(v) for v in re.findall(r'\by="(-?[\d.]+)"', inner)]
    for d in re.findall(r'\bd="([^"]+)"', inner):
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
        xs += nums[0::2]
        ys += nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def column_labels(n_slices: int) -> list[str]:
    """reset, CX layer 1..k, readout (readout only if it is the last column)."""
    labels = ["reset"] + [f"CX layer {i}" for i in range(1, n_slices - 1)]
    labels.append("readout" if INCLUDE_READOUT else f"CX layer {n_slices - 1}")
    return labels


def main() -> None:
    circuits = {b: build_circuit(x_basis=(b == "X")) for b in BASES}
    any_circ = next(iter(circuits.values()))
    n_slices = any_circ.num_ticks + (1 if INCLUDE_READOUT else 0)
    labels = column_labels(any_circ.num_ticks + 1)[:n_slices]

    # Collect every panel, then crop them all to a common content bbox so the
    # lattice stays aligned across ticks and rows.
    panels = {
        (b, c): panel_inner(circuits[b], c)
        for b in BASES
        for c in range(n_slices)
    }
    boxes = [content_bbox(p) for p in panels.values()]
    min_x = min(bb[0] for bb in boxes) - CROP_PAD
    min_y = min(bb[1] for bb in boxes) - CROP_PAD
    max_x = max(bb[2] for bb in boxes) + CROP_PAD
    max_y = max(bb[3] for bb in boxes) + CROP_PAD
    vb_w, vb_h = max_x - min_x, max_y - min_y
    view_box = f"{min_x:.1f} {min_y:.1f} {vb_w:.1f} {vb_h:.1f}"
    cell_w = CELL_H * vb_w / vb_h

    title = TITLE
    if title is None:
        state = "|+>_L" if PLUS else "|0>_L"
        title = f"{state}  {DX}x{DZ}  {METHOD} cascade — stabilizer construction"
    title_h = 28.0 if title else 0.0
    caption_h = 18.0
    top = PAD + title_h + caption_h

    n_rows = len(BASES)
    width = LEFT + n_slices * cell_w + (n_slices - 1) * COL_GAP + PAD
    height = top + n_rows * CELL_H + (n_rows - 1) * ROW_GAP + PAD

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.1f} {height:.1f}" font-family="sans-serif">',
        f'<rect width="{width:.1f}" height="{height:.1f}" fill="white"/>',
    ]

    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{PAD + 19:.1f}" text-anchor="middle" '
            f'font-size="17">{title}</text>'
        )

    for c, label in enumerate(labels):
        cx = LEFT + c * (cell_w + COL_GAP) + cell_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{top - 5:.1f}" text-anchor="middle" '
            f'font-size="14">{label}</text>'
        )

    for r, basis in enumerate(BASES):
        row_y = top + r * (CELL_H + ROW_GAP)
        ly = row_y + CELL_H / 2
        lx = LEFT - 36
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 {lx:.1f} {ly:.1f})">{basis} stabilizers</text>'
        )
        for c in range(n_slices):
            cx = LEFT + c * (cell_w + COL_GAP)
            parts.append(
                f'<svg x="{cx:.1f}" y="{row_y:.1f}" width="{cell_w:.1f}" '
                f'height="{CELL_H:.1f}" viewBox="{view_box}" '
                f'preserveAspectRatio="xMidYMid meet">{panels[(basis, c)]}</svg>'
            )

    parts.append("</svg>")
    OUT_PATH.write_text("\n".join(parts))
    print(f"Saved {OUT_PATH}  ({n_rows} rows x {n_slices} cols)")


if __name__ == "__main__":
    main()
