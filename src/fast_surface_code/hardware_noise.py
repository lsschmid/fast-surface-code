"""Hardware-realistic (neutral-atom / QuEra Gemini) noise for stim circuits.

This bridges a stim *preparation* circuit through ``bloqade-circuit`` to attach
the Gemini device error model, then lowers the result back to stim for fast
Clifford sampling. Concretely, a prep circuit is:

1. converted to a cirq circuit (``stimcirq``),
2. transpiled to the neutral-atom native gate set (CZ + PhasedXZ) by bloqade,
3. annotated with the Gemini Pauli-error channels — per single-qubit gate, the
   correlated CZ channel, and a *move/idle* channel — by a bloqade noise model,
4. lowered back to a stim circuit (``stimcirq``).

Move model: we use bloqade's *one-zone* Gemini model, which applies a single,
fixed ``mover``/``sitter`` Pauli channel for every move — each CZ's control atom
moves to its target (a nearest-neighbour hop) and everyone else sits. There is
no routing and no geometry: the same move-error is used for every move,
independent of direction or distance. (The two-zone model routes atoms between
a storage and an entangling zone; that routing is combinatorial and becomes
prohibitively slow beyond d=3, so it is not the default.)

The Gemini rates are Z-biased (dephasing dominates), which is exactly the error
type the unitary cascade spreads — so this noise model exercises the regime the
preparation is *not* protected against.

Requires the optional ``hardware`` dependency group:  ``uv sync --group hardware``
"""

from __future__ import annotations

import stim

try:
    import cirq
    import stimcirq
    from bloqade.cirq_utils.noise import (
        GeminiOneZoneNoiseModel,
        GeminiTwoZoneNoiseModel,
    )
    from bloqade.cirq_utils.parallelize import transpile
except ImportError as exc:  # pragma: no cover - clearer error for missing extra
    raise ImportError(
        "hardware_noise requires the 'hardware' dependency group. "
        "Install it with:  uv sync --group hardware"
    ) from exc


def _snap(x: float, step: float = 0.5) -> float:
    """Round a gate exponent to the nearest Clifford value."""
    return round(x / step) * step


def _snap_to_clifford(circuit: cirq.Circuit) -> cirq.Circuit:
    """Snap native-gate angles to exact Cliffords so stimcirq can convert them.

    bloqade's native gate set emits ``PhasedXZGate``/``CZPowGate`` with tiny
    floating-point angle error (e.g. 0.4999… for 0.5). stimcirq only translates
    exact Cliffords, so we snap every angle to the nearest multiple of 0.5.
    Noise channels are left untouched.
    """

    def fix(op: cirq.Operation) -> cirq.Operation:
        gate = op.gate
        if isinstance(gate, cirq.PhasedXZGate):
            return cirq.PhasedXZGate(
                axis_phase_exponent=_snap(gate.axis_phase_exponent),
                x_exponent=_snap(gate.x_exponent),
                z_exponent=_snap(gate.z_exponent),
            ).on(*op.qubits)
        if isinstance(gate, cirq.CZPowGate):
            return cirq.CZ.on(*op.qubits)
        return op

    return cirq.Circuit(
        cirq.Moment(fix(op) for op in moment.operations) for moment in circuit
    )


def _strip_annotations(circuit: stim.Circuit) -> stim.Circuit:
    """Drop instructions cirq can't represent (QUBIT_COORDS, TICK)."""
    out = stim.Circuit()
    for inst in circuit:
        if inst.name in ("QUBIT_COORDS", "TICK"):
            continue
        out.append(inst)
    return out


_ZONE_MODELS = {"one": GeminiOneZoneNoiseModel, "two": GeminiTwoZoneNoiseModel}


def apply_gemini_noise(
    prep: stim.Circuit,
    *,
    zone: str = "one",
    scaling_factor: float = 1.0,
) -> stim.Circuit:
    """Attach the Gemini hardware noise model to a stim *preparation* circuit.

    The prep circuit must contain only resets and unitary gates (R/RX/H/CX/CZ)
    — no measurements. Qubit indices are preserved, so measurement, detector,
    and observable instructions built on the original indices can be appended to
    the returned circuit afterwards.

    Args:
        prep: the noiseless preparation circuit (resets + unitary gates only).
        zone: ``"one"`` (default; all atoms in the entangling zone, each CZ's
            control makes one nearest-neighbour move — a fixed move-error per
            gate, no routing) or ``"two"`` (storage + entangling zones, routes
            atoms — prohibitively slow beyond d=3).
        scaling_factor: multiplies every Pauli rate. 1.0 = device defaults,
            0.0 = noiseless, 2.0 = double. Use this to sweep noise strength.

    Returns:
        A stim circuit implementing the prep in native gates with Gemini Pauli
        noise channels inserted.
    """
    if zone not in _ZONE_MODELS:
        raise ValueError(f"zone must be one of {sorted(_ZONE_MODELS)}, got {zone!r}")
    model = _ZONE_MODELS[zone](scaling_factor=scaling_factor)

    cirq_prep = stimcirq.stim_circuit_to_cirq_circuit(_strip_annotations(prep))
    system_qubits = sorted(cirq_prep.all_qubits())
    native = transpile(cirq_prep)

    noisy = cirq.Circuit()
    for op_tree in model.noisy_moments(native.moments, system_qubits):
        noisy += cirq.Circuit(op_tree)

    # Preserve the original (sparse) qubit indices so that measurement,
    # detector, and observable instructions built on those indices can be
    # appended to the result. Without an explicit map, stimcirq compacts the
    # qubits to 0..n-1 and the record targets no longer line up.
    n_qubits = prep.num_qubits
    qubit_to_index = {cirq.LineQubit(i): i for i in range(n_qubits)}
    return stimcirq.cirq_circuit_to_stim_circuit(
        _snap_to_clifford(noisy), qubit_to_index_dict=qubit_to_index
    )
