# Fast Surface Code

Unitary, measurement-free preparation of surface-code logical states, and a
memory-experiment evaluation of their logical error rate under circuit-level
noise.

A patch is prepared in `|0⟩_L` or `|+⟩_L` by a stabilizer-expanding CNOT
cascade — either **unidirectional** (`O(d)` depth) or **bidirectional /
middle-out** (`O(d/2)` depth). The preparation uses no ancilla qubits and no
mid-circuit measurements, and is fault-tolerant to distance `d` against the
error type it expands.

Patches may be **square or rectangular**. Size is set by two independent
distances: `dx` (X-distance = number of rows) and `dz` (Z-distance = number of
columns), both odd and ≥ 3:

```python
SurfaceCode(d=7)          # square 7x7
SurfaceCode(dx=3, dz=7)   # rectangular: X-distance 3, Z-distance 7
```

For a rectangular patch the `|0⟩` memory fails via a logical X (crosses the
rows → distance `dx`) and `|+⟩` fails via a logical Z (crosses the columns →
distance `dz`); the two cascade depths follow the swept dimension (`dz` for
`|0⟩`, `dx` for `|+⟩`).

## Project structure

```
src/fast_surface_code/
    surface_code.py   # SurfaceCode patch: layout, unitary init, measurement, detectors
    noise.py          # CircuitLevelNoise: depolarizing circuit-level noise
    hardware_noise.py # Gemini (neutral-atom) noise via bloqade-circuit [optional]

evaluation/
    eval_zero_state.py  # Memory experiment: prepare -> measure -> decode -> logical error rate
    plot_results.py     # Plot logical vs physical error rate per distance

experiments/
    view_circuits.py    # Build the prep circuits and open them in Crumble to inspect

tests/
    test_surface_code.py
```

## The experiment

`eval_zero_state.py` builds, for each `(shape, strength, method, basis)`:

1. **Prepare** the state — `unitary_init_rows()` (`rows`, `O(d)` parallel cascade),
   `unitary_init_bidirectional()` (`bidirectional`, `O(d/2)` middle-out),
   `unitary_init_sequential()` (`sequential`, the single-pivot `3(d-1)` construction
   of arXiv:2601.05113), or `ceil(d/2)` measurement rounds (`measurement`, baseline).
2. **Measure** all data qubits in the matching basis (`M` for `|0⟩`, `MX` for `|+⟩`).
3. **Add the logical observable** (`Z_L` for `|0⟩`, `X_L` for `|+⟩`).
4. **Decode** with pymatching on the circuit DEM and record the logical error rate
   (adaptive sampling until `MIN_ERRORS` failures).

### Noise models

Set `NOISE_MODEL` at the top of `eval_zero_state.py`:

- `"depolarizing"` — uniform circuit-level depolarizing noise (`CircuitLevelNoise`),
  swept over the physical error rate `p`.
- `"gemini"` — the neutral-atom QuEra Gemini hardware model via `bloqade-circuit`:
  the prep is compiled to native gates (CZ + PhasedXZ) and annotated with the
  device's Z-biased Pauli-error channels per gate, plus a single fixed
  move/idle channel for every move (nearest-neighbour, no routing — the
  `"one"`-zone model). Swept over a noise scaling factor (`1.0` = device defaults).
  Requires the optional dependency group:

  ```bash
  uv sync --group hardware
  ```

## Quick start

```bash
# Run the sweep (edit DISTANCES / P_VALUES / METHODS / BASES at the top of the file)
uv run python3 evaluation/eval_zero_state.py   # -> evaluation/results.json

# Plot logical error rate vs physical error rate, per method and distance
uv run python3 evaluation/plot_results.py

# Inspect the prep circuits in Crumble (edit the parameter block at the top)
uv run python3 experiments/view_circuits.py

# Tests
uv run python3 -m pytest tests/ -q
```

## Reproducing the paper figures

The workshop paper (QCE26 / CEVNAC 2026 — source in `paper/`) uses the
following figure pipeline. `evaluation/results.json` contains the exact
simulation data behind the benchmark plot, so Fig. 4 can be reproduced
without re-running the sweep.

| Paper figure | Script | Output |
|---|---|---|
| Fig. 1 (patch overview)        | `evaluation/plot_patch_overview.py`   | `patch_overview.pdf` |
| Fig. 2a (stabilizer growth)    | `evaluation/plot_stab_slices.py`      | `stab_slices.svg` (convert with `rsvg-convert`) |
| Fig. 2b (AOD moves)            | `evaluation/plot_aod_moves.py`        | `aod_moves.pdf` |
| Fig. 3 (error propagation)     | `evaluation/plot_error_prop_cnots.py` | `error_prop_{X,Z}.pdf` |
| Fig. 4 (logical error rates)   | `evaluation/plot_results.py`          | `plot.pdf` (reads `results.json`) |
| Tab. II (noise rates)          | values from the `bloqade` one-zone noise model |

To regenerate `results.json` from scratch (hours of runtime at the paper's
shot budget), run `evaluation/eval_zero_state.py` with `NOISE_MODEL =
"gemini"`; see `paper/README.md` for the figure export commands.

## License

MIT — see [LICENSE](LICENSE). If you use this code, please cite the paper
(see [CITATION.cff](CITATION.cff)).
