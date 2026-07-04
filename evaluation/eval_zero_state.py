"""Evaluate logical error rate for |0⟩ and |+⟩ states under a chosen noise model.

Compares initialization methods (unitary rows / bidirectional / sequential
cascades, and a measurement-based baseline) under one of two noise models:

  - "depolarizing": uniform circuit-level depolarizing noise (CircuitLevelNoise),
    swept over the physical error rate p.
  - "gemini": the neutral-atom QuEra Gemini hardware model (native CZ + PhasedXZ
    gates, Z-biased Pauli channels per gate, and a single nearest-neighbour
    move/idle channel for every move), via bloqade-circuit. Swept over a noise
    "scaling factor" (1.0 = device defaults). Requires:  uv sync --group hardware

Sweeps over patch shapes, a noise-strength axis, and basis (plus=False/True),
decodes with pymatching, and saves results to results.json.

Uses adaptive sampling: samples in batches and stops early once enough errors
are collected for a reliable estimate, or the max budget is exhausted.
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_surface_code.noise import CircuitLevelNoise
from fast_surface_code.surface_code import SurfaceCode

# --- Parameters ---
# Noise model: "depolarizing" (uniform circuit-level) or "gemini" (neutral-atom
# QuEra Gemini hardware model via bloqade-circuit; needs `uv sync --group hardware`).
NOISE_MODEL = "depolarizing"
# gemini only: "one" (default; fixed move-error per CZ, no routing) or "two"
# (storage+entangling zones — routes atoms, prohibitively slow beyond d=3).
GEMINI_ZONE = "one"

# Patch shapes as (dx, dz): dx = X-distance = #rows, dz = Z-distance = #cols.
# Use equal entries for square patches, e.g. (5, 5); differ for rectangular.
SHAPES = [(3, 3), (5, 5), (7, 7)]

# Noise-strength axis. For "depolarizing" these are physical error rates p; for
# "gemini" they are noise scaling factors (1.0 = device defaults).
P_VALUES = np.logspace(-4, -2, 10).tolist()
SCALING_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]

# "sequential" is the paper's single-pivot O(d) construction (both bases).
# The gemini model does not support the mid-circuit-measurement "measurement"
# method; those combinations are skipped.
METHODS = ["rows", "bidirectional", "sequential", "measurement"]
BASES = [False, True]  # plus=False (|0⟩, Z logical), plus=True (|+⟩, X logical)

# Sampling: sample in batches until MIN_ERRORS failures are collected.
MIN_ERRORS = 200
BATCH_SIZE = 50_000
MAX_SHOTS = 10_000_000  # safety cap

MAX_WORKERS = None  # None = number of CPUs
# ------------------


def _build_prepared_code(dx: int, dz: int, method: str, plus: bool) -> SurfaceCode:
    """Build a prepared, measured code with detectors and the logical observable.

    Returns the SurfaceCode whose ``_pre_final_circuit`` is the preparation
    (resets + unitary gates) and whose ``_circuit`` additionally contains the
    final measurement, detectors, and observable.
    """
    code = SurfaceCode(dx=dx, dz=dz, plus=plus)

    if method == "rows":
        code.unitary_init_rows()
    elif method == "bidirectional":
        code.unitary_init_bidirectional()
    elif method == "sequential":
        code.unitary_init_sequential()
    elif method == "measurement":
        # Initialize data qubits in the matching basis so first-round
        # stabilizers are deterministic.
        reset_op = "RX" if plus else "R"
        code._circuit.append(reset_op, code.qubits)
        code._circuit.append("TICK")
        n_rounds = math.ceil(max(dx, dz) / 2)
        for _ in range(n_rounds):
            code.measure_round()
    else:
        raise ValueError(f"Unknown method: {method}")

    code.final_measurement()

    # Append OBSERVABLE_INCLUDE for the logical operator
    obs = code.get_observable()
    n_data = len(code.qubits)
    for i, val in enumerate(obs):
        if val == 1:
            code._circuit.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-(n_data - i))], [0]
            )

    return code


def build_circuit(dx: int, dz: int, method: str, plus: bool) -> stim.Circuit:
    """Build an ideal circuit with detectors and observable."""
    return _build_prepared_code(dx, dz, method, plus)._circuit


def build_noisy_circuit(
    dx: int, dz: int, p: float, method: str, plus: bool
) -> stim.Circuit:
    """Build the noisy circuit for the configured noise model.

    ``p`` is the noise-strength axis value: the physical error rate for the
    depolarizing model, or the noise scaling factor for the gemini model.
    """
    if NOISE_MODEL == "depolarizing":
        circuit = build_circuit(dx, dz, method, plus)
        noise = CircuitLevelNoise(p_tqg=p, p_sqg=p, p_meas=p, p_init=p)
        return noise.apply(circuit)

    if NOISE_MODEL == "gemini":
        # Imported lazily so depolarizing runs don't need the hardware extra.
        from fast_surface_code.hardware_noise import apply_gemini_noise

        code = _build_prepared_code(dx, dz, method, plus)
        prep = code._pre_final_circuit
        tail = code._circuit[len(prep):]
        return apply_gemini_noise(prep, zone=GEMINI_ZONE, scaling_factor=p) + tail

    raise ValueError(f"Unknown NOISE_MODEL: {NOISE_MODEL!r}")


def run_single(dx: int, dz: int, p: float, method: str, plus: bool) -> dict:
    """Run one (dx, dz, p, method, plus) simulation with adaptive sampling.

    ``p`` is the noise-strength axis value: the physical error rate for the
    depolarizing model, or the noise scaling factor for the gemini model.
    """
    noisy = build_noisy_circuit(dx, dz, p, method, plus)

    # The gemini model emits a correlated two-qubit Pauli channel for the CZ,
    # which stim only turns into a DEM with approximate_disjoint_errors=True.
    dem = noisy.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = noisy.compile_detector_sampler()

    total_shots = 0
    total_errors = 0

    while total_shots < MAX_SHOTS:
        batch = min(BATCH_SIZE, MAX_SHOTS - total_shots)
        syndromes, obs_flips = sampler.sample(batch, separate_observables=True)
        predictions = matcher.decode_batch(syndromes)
        total_errors += int(np.sum(predictions != obs_flips))
        total_shots += batch

        if total_errors >= MIN_ERRORS:
            break

    rate = total_errors / total_shots

    # Wilson score interval for binomial proportion
    z = 1.96  # 95% confidence
    n = total_shots
    p_hat = rate
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_w = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0, center - half_w)
    ci_high = center + half_w

    basis = "plus" if plus else "zero"
    return {
        "dx": dx,
        "dz": dz,
        "shape": f"{dx}x{dz}",
        "p": p,
        "noise_model": NOISE_MODEL,
        "method": method,
        "basis": basis,
        "shots": total_shots,
        "errors": total_errors,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    strengths = SCALING_FACTORS if NOISE_MODEL == "gemini" else P_VALUES
    methods = METHODS
    if NOISE_MODEL == "gemini" and "measurement" in methods:
        # The gemini path splices a noiseless measurement tail onto the noisy
        # preparation, which requires a measurement-free prep.
        methods = [m for m in methods if m != "measurement"]
        print("gemini noise: skipping the 'measurement' method (mid-circuit MR).")

    params = list(product(SHAPES, strengths, methods, BASES))
    axis = "scaling" if NOISE_MODEL == "gemini" else "p"
    print(
        f"Running {len(params)} simulations "
        f"[noise={NOISE_MODEL}, axis={axis}] (min {MIN_ERRORS} errors each) ..."
    )

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single, dx, dz, p, method, plus): (dx, dz, p, method, plus)
            for (dx, dz), p, method, plus in params
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            dx, dz, p, method, plus = futures[future]
            basis = "|+⟩" if plus else "|0⟩"
            print(
                f"  [{i}/{len(params)}] {basis} {dx}x{dz} {axis}={p:g} {method}: "
                f"errors={result['errors']}/{result['shots']:,} "
                f"rate={result['rate']:.4e}"
            )

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
