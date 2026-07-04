# Figure notes / draft captions

Working captions and provenance for the paper figures. Colors follow the stim
detector-slice convention throughout: **X-type = red, Z-type = blue**.

Generated PDFs live in `../evaluation/` (build artifacts, untracked). Export the
ones you use into `figures/`.

---

## Fig. — Stabilizer construction (`evaluation/plot_stab_slices.py` → `stab_slices.svg`)

Detector-slice view of the bidirectional cascade preparing `|+>_L` on a 5×3
patch. Top row: Z-type stabilizers (blue); bottom row: X-type stabilizers (red).
One column per circuit tick (reset → CX layers → readout); each plaquette grows
as the middle-out cascade attaches data qubits, until the full-weight
stabilizers are formed.

## Fig. — AOD atom moves (`evaluation/plot_aod_moves.py` → `aod_moves.pdf`)

The cascade realized as neutral-atom moves. One panel per layer: a dashed ghost
marks each vacated site, an arrow the one-lattice-step hop, and green bonds the
CZ pairs. Each cascade layer is a single collective nearest-neighbour move (a
whole row shifting to its neighbour); the middle seed row is picked up once and
serves both neighbours, and fresh outer rows are shuttled inward to the
stationary built block — an AOD-native schedule with no routing.

## Table — Preparation depth (`Table` in `main.tex`, `tab:depth`)

Entangling-gate layers (= collective AOD moves) vs. distance for the three
constructions: single-pivot `3(d-1)`, unidirectional `d`, bidirectional
`(d+3)/2`. The bidirectional cascade roughly halves the depth of the O(d)
baselines at every distance (near-/mid-/large-scale d = 3/15/21).

## Fig. — Error propagation, protected direction (`error_prop_X.pdf`)

Propagation of two single X (bit-flip) faults **(a)** and **(b)** through the
preparation cascade of a `|0>_L` patch, one panel per CNOT layer (propagating
CNOTs highlighted red, error support in red; stabilizer surfaces and the logical
directions `X_L`, `Z_L` shown once for reference). X is the *protected* error
type: **(a)** spreads into a stabilizer — equivalent to no error; **(b)** spreads
to a weight-2 error orthogonal to the logical — correctable. A single X fault
never propagates to a logical operator, so preparation is fault tolerant against
this error type (distance `d`).

## Fig. — Error propagation, conjugate direction (`error_prop_Z.pdf`)

Same as above for two single Z (phase) faults **(a)** and **(b)**; Z is the
*conjugate* type, which spreads **along** the logical `Z_L`. **(a)** spreads to
the full logical `Z_L`, which stabilizes `|0>_L` — equivalent to no error;
**(b)** spreads to an irreducible weight-2 error along the logical. Because a
single fault can produce a weight-2 error, the preparation is **not** fault
tolerant in the conjugate direction. This is harmless for a stored `|0>_L`
(a phase error commutes with the state) but disqualifies the patch as a
Steane-type logical ancilla, where the spread error would propagate onto the
data.
