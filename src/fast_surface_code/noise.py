"""Minimal circuit-level depolarizing noise for stim circuits.

Self-contained equivalent of ``mqt.qecc.circuit_synthesis.noise.CircuitLevelNoise``:

- after every reset (``R``/``RX``/...):        DEPOLARIZE1(p_init)
- after every single-qubit gate (``H``/...):   DEPOLARIZE1(p_sqg)
- after every two-qubit gate (``CX``/...):     DEPOLARIZE2(p_tqg)  (per gate pair)
- measure+reset (``MR``/...):                  measurement flip p_meas

Note: plain measurements (``M``/``MX``) are left noiseless, matching the
reference model. For destructive readout at the end of a memory experiment
this is intentional — the readout basis is measured once and discarded.
"""

from __future__ import annotations

import stim

# stim gate classification (subset that this project emits + common aliases)
_SQGS = {
    "H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
    "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG",
    "C_XYZ", "C_ZYX", "H_XY", "H_XZ", "H_YZ",
}
_TQGS = {
    "CNOT", "CX", "CY", "CZ", "XCX", "XCZ", "ZCX", "ZCZ",
    "SWAP", "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_YY", "SQRT_ZZ",
}
_RESETS = {"R", "RX", "RY", "RZ"}
_MEASURE_RESETS = {"MR", "MRX", "MRY", "MRZ"}


class CircuitLevelNoise:
    """Circuit-level depolarizing noise model."""

    def __init__(
        self,
        p_tqg: float,
        p_sqg: float,
        p_meas: float,
        p_init: float,
    ) -> None:
        self.p_tqg = p_tqg
        self.p_sqg = p_sqg
        self.p_meas = p_meas
        self.p_init = p_init

    def apply(self, circ: stim.Circuit) -> stim.Circuit:
        """Return a copy of ``circ`` with depolarizing noise inserted."""
        noisy = stim.Circuit()
        for op in circ:
            name = op.name
            if name in _SQGS:
                for targets in op.target_groups():
                    noisy.append_operation(name, targets)
                    noisy.append_operation(
                        "DEPOLARIZE1", [t.qubit_value for t in targets], self.p_sqg
                    )
            elif name in _RESETS:
                for targets in op.target_groups():
                    noisy.append_operation(name, targets)
                    noisy.append_operation(
                        "DEPOLARIZE1", [t.qubit_value for t in targets], self.p_init
                    )
            elif name in _TQGS:
                for targets in op.target_groups():
                    noisy.append_operation(name, targets)
                    noisy.append_operation(
                        "DEPOLARIZE2", [t.qubit_value for t in targets], self.p_tqg
                    )
            elif name in _MEASURE_RESETS:
                for targets in op.target_groups():
                    noisy.append_operation(name, targets, self.p_meas)
            else:
                noisy.append_operation(op)
        return noisy
