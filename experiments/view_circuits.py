"""Build state-preparation circuits and open them in Crumble for inspection.

Crumble is stim's browser-based circuit viewer (algassert.com/crumble). Each
selected (state, method) circuit is opened in a new browser tab; the noiseless
circuit is used so the layout, cascade order, and detectors are easy to read.

Run:  uv run python3 experiments/view_circuits.py
"""

from __future__ import annotations

import math
import sys
import webbrowser
from pathlib import Path

import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.surface_code import SurfaceCode

# ======================================================================
# Parameters — edit these
# ======================================================================
DX = 5                                  # X-distance = #rows   (odd, >= 3)
DZ = 7                                  # Z-distance = #cols   (odd, >= 3)  (DX != DZ = rectangular)
STATES = ["zero", "plus"]                            # "zero" (|0>_L), "plus" (|+>_L)
METHODS = ["rows", "bidirectional", "sequential"]    # + "measurement" baseline
INCLUDE_FINAL_MEASUREMENT = True        # append destructive readout + detectors
INCLUDE_OBSERVABLE = False              # add the logical observable
OPEN_IN_BROWSER = True                  # False -> just print the Crumble URLs
# ======================================================================


def build_circuit(dx: int, dz: int, state: str, method: str) -> stim.Circuit:
    """Build one noiseless preparation circuit."""
    plus = state == "plus"
    code = SurfaceCode(dx=dx, dz=dz, plus=plus)

    if method == "rows":
        code.unitary_init_rows()
    elif method == "bidirectional":
        code.unitary_init_bidirectional()
    elif method == "sequential":
        code.unitary_init_sequential()
    elif method == "measurement":
        reset_op = "RX" if plus else "R"
        code._circuit.append(reset_op, code.qubits)
        code._circuit.append("TICK")
        for _ in range(math.ceil(max(dx, dz) / 2)):
            code.measure_round()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if INCLUDE_FINAL_MEASUREMENT:
        code.final_measurement()

    if INCLUDE_OBSERVABLE:
        obs = code.get_observable()
        n_data = len(code.qubits)
        for i, val in enumerate(obs):
            if val == 1:
                code._circuit.append(
                    "OBSERVABLE_INCLUDE", [stim.target_rec(-(n_data - i))], [0]
                )

    return code._circuit


def main() -> None:
    for state in STATES:
        for method in METHODS:
            circ = build_circuit(DX, DZ, state, method)
            label = f"|{'+' if state == 'plus' else '0'}>  {DX}x{DZ}  {method}"
            url = circ.to_crumble_url()
            print(f"{label}: {circ.num_qubits} qubits, depth {circ.num_ticks} ticks")
            print(f"  {url}\n")
            if OPEN_IN_BROWSER:
                webbrowser.open(url)


if __name__ == "__main__":
    main()
