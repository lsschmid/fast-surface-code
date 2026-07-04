from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim

if TYPE_CHECKING:
    from fast_surface_code.noise import CircuitLevelNoise as NoiseModel


class SurfaceCode:
    """A single d×d surface code patch.

    Qubit layout:
    - Integer row r (r=0..d-1): d qubits at cols 0..d-1.
    - Half-integer row r+0.5 (r=0..d-2): measure qubits for stabilizer checks.
    - Indices assigned sequentially: row 0, row 0.5, row 1, row 1.5, ...

    The _circuit property accumulates operations as init methods are called.
    It starts with QUBIT_COORDS and R on all data qubits.
    """

    def __init__(
        self,
        d: int | None = None,
        origin: tuple[int, int] = (0, 0),
        plus: bool = False,
        *,
        dx: int | None = None,
        dz: int | None = None,
    ) -> None:
        """Create a surface code patch.

        Size is either square (pass ``d``) or rectangular (pass ``dx`` and
        ``dz``).  ``dx`` is the X-distance = number of data *rows*; ``dz`` is
        the Z-distance = number of data *columns*.  Both must be odd and >= 3.
        """
        if d is not None:
            if dx is not None or dz is not None:
                raise ValueError("Pass either d (square) or dx/dz, not both.")
            dx = dz = d
        if dx is None or dz is None:
            raise ValueError("Provide d (square) or both dx and dz (rectangular).")
        assert dx >= 3 and dx % 2 == 1, "dx must be odd and >= 3"
        assert dz >= 3 and dz % 2 == 1, "dz must be odd and >= 3"
        self.d_rows = dx  # X-distance (number of data rows)
        self.d_cols = dz  # Z-distance (number of data columns)
        self.origin = origin
        self.plus = plus
        self._build_layout()
        self._init_circuit()

    @property
    def d(self) -> int:
        """Distance shorthand for square patches (raises when rectangular)."""
        if self.d_rows != self.d_cols:
            raise AttributeError(
                "Rectangular patch has no single 'd'; use d_rows / d_cols."
            )
        return self.d_rows

    def _build_layout(self) -> None:
        d_rows, d_cols = self.d_rows, self.d_cols
        row_off, col_off = self.origin

        self.qubits: list[int] = []
        self.measure_qubits: list[int] = []
        self.z_measure_qubits: list[int] = []
        self.qubit_coords: dict[int, tuple[float, float]] = {}
        self._grid: dict[tuple[int, int], int] = {}
        self.stabilizers: dict[int, list[int]] = {}
        self._row_start: list[int] = []

        idx = 0
        for r in range(d_rows):
            self._row_start.append(idx)
            for c in range(d_cols):
                self.qubit_coords[idx] = (r + row_off, c + col_off)
                self._grid[(r, c)] = idx
                self.qubits.append(idx)
                idx += 1

            if r < d_rows - 1:
                for mc in self._measure_cols(r):
                    c_int = int(mc)
                    # Place boundary weight-2 X-mq at the boundary edge
                    # to avoid coordinate overlap with Z-mq at the same face.
                    if r == 0 and c_int % 2 == 1:
                        row_coord = -0.5 + row_off
                    elif r == d_rows - 2 and c_int % 2 == 0:
                        row_coord = d_rows - 0.5 + row_off
                    else:
                        row_coord = r + 0.5 + row_off
                    self.qubit_coords[idx] = (row_coord, mc + col_off)
                    self.measure_qubits.append(idx)
                    idx += 1

        # Z-type ancilla qubits (added after all X-type qubits)
        for r in range(d_rows - 1):
            # Interior Z-type: at (r+0.5, c+0.5) where r+c is odd
            for c in range(d_cols - 1):
                if (r + c) % 2 == 1:
                    self.qubit_coords[idx] = (r + 0.5 + row_off, c + 0.5 + col_off)
                    self.z_measure_qubits.append(idx)
                    idx += 1
            # Left boundary Z-type: (r+0.5, -0.5) for even r
            if r % 2 == 0:
                self.qubit_coords[idx] = (r + 0.5 + row_off, -0.5 + col_off)
                self.z_measure_qubits.append(idx)
                idx += 1
            # Right boundary Z-type: (r+0.5, d_cols-0.5) for odd r
            if r % 2 == 1:
                self.qubit_coords[idx] = (r + 0.5 + row_off, d_cols - 0.5 + col_off)
                self.z_measure_qubits.append(idx)
                idx += 1

        self.total_qubits = idx
        self._build_stabilizers()
        self._build_z_stabilizers()

        # Invariant: a valid odd×odd rotated patch encodes 1 logical qubit, so
        # the stabilizer count must be n_data - 1.  (For rectangular patches the
        # X and Z counts are individually asymmetric; only the sum is fixed.)
        # A boundary bug almost always breaks this.
        n_stab = len(self.measure_qubits) + len(self.z_measure_qubits)
        assert n_stab == d_rows * d_cols - 1, (
            f"stabilizer count {n_stab} != {d_rows * d_cols - 1} for {d_rows}x{d_cols} "
            f"(X={len(self.measure_qubits)}, Z={len(self.z_measure_qubits)})"
        )

    def _init_circuit(self) -> None:
        """Initialize _circuit with QUBIT_COORDS (no data qubit reset — init methods handle that)."""
        self._circuit = stim.Circuit()
        self._total_measurements = 0
        self._mq_last_rec: dict[int, int] = {}  # measure qubit → absolute record index
        self._measured_pairs: set[int] = set()     # half-pair indices that have been measurement-initialized
        self._unitary_pairs: set[int] = set()     # half-pair indices that have been unitarily initialized
        self._detector_defs: list[list[int]] = []  # each entry: abs measurement indices for one detector
        for qidx in sorted(self.qubit_coords):
            vc = self._visual_coords(qidx)
            self._circuit.append("QUBIT_COORDS", [qidx], list(vc))

    def _measure_cols(self, r: int) -> list[float]:
        """Column positions for measure qubits in half-row r+0.5.

        Boundary rows (r=0, r=d-2): all d-1 positions (0.5, 1.5, …).
        Interior rows: staggered pattern — odd rows start at 1.5,
        even rows start at 0.5, spacing 2.
        """
        if r == 0 or r == self.d_rows - 2:
            return [k + 0.5 for k in range(self.d_cols - 1)]
        elif r % 2 == 1:
            return [1.5 + 2 * k for k in range((self.d_cols - 1) // 2)]
        else:
            return [0.5 + 2 * k for k in range((self.d_cols - 1) // 2)]

    def _build_stabilizers(self) -> None:
        """For each measure qubit, find its neighboring patch qubits."""
        for mq in self.measure_qubits:
            row_f, col_f = self.qubit_coords[mq]
            r = int(row_f - self.origin[0])
            c_int = int(col_f - self.origin[1])

            connect_up = True
            connect_down = True
            if r == 0 and c_int % 2 == 1:
                connect_down = False
            elif r == self.d_rows - 2 and c_int % 2 == 0:
                connect_up = False

            neighbors = []
            if connect_up:
                for dc in [0, 1]:
                    key = (r, c_int + dc)
                    if key in self._grid:
                        neighbors.append(self._grid[key])
            if connect_down:
                for dc in [0, 1]:
                    key = (r + 1, c_int + dc)
                    if key in self._grid:
                        neighbors.append(self._grid[key])

            self.stabilizers[mq] = neighbors

    def _build_z_stabilizers(self) -> None:
        """Build Z-type stabilizer plaquettes using physical Z-type ancilla qubits.

        Z-type stabilizers are the dual of X-type:
        - Weight-4 blocks at centers (r+0.5, c+0.5) where r+c is odd.
        - Weight-2 at left boundary (col 0): vertical pairs for even r.
        - Weight-2 at right boundary (col d-1): vertical pairs for odd r.

        Stored as ``self.z_stabilizers``: maps Z-type ancilla qubit index →
        list of data qubit indices (matching the pattern of ``self.stabilizers``).
        """
        d_rows, d_cols = self.d_rows, self.d_cols
        row_off, col_off = self.origin
        self.z_stabilizers: dict[int, list[int]] = {}

        # Build coord → Z-ancilla index lookup
        z_coord_to_idx = {}
        for zmq in self.z_measure_qubits:
            z_coord_to_idx[self.qubit_coords[zmq]] = zmq

        # Interior weight-4: centers where r+c is odd
        for r in range(d_rows - 1):
            for c in range(d_cols - 1):
                if (r + c) % 2 == 1:
                    coord = (r + 0.5 + row_off, c + 0.5 + col_off)
                    zmq = z_coord_to_idx[coord]
                    self.z_stabilizers[zmq] = [
                        self._grid[(r, c)], self._grid[(r, c + 1)],
                        self._grid[(r + 1, c)], self._grid[(r + 1, c + 1)],
                    ]

        # Weight-2 at left boundary (col 0): r even
        for r in range(0, d_rows - 1, 2):
            coord = (r + 0.5 + row_off, -0.5 + col_off)
            zmq = z_coord_to_idx[coord]
            self.z_stabilizers[zmq] = [
                self._grid[(r, 0)], self._grid[(r + 1, 0)],
            ]

        # Weight-2 at right boundary (col d_cols-1): r odd
        for r in range(1, d_rows - 1, 2):
            coord = (r + 0.5 + row_off, d_cols - 0.5 + col_off)
            zmq = z_coord_to_idx[coord]
            self.z_stabilizers[zmq] = [
                self._grid[(r, d_cols - 1)], self._grid[(r + 1, d_cols - 1)],
            ]

    def _visual_coords(self, qidx: int) -> tuple[float, float]:
        """Return (col, row) for stim emission — swapped so rows appear horizontal in Crumble.

        Adds a (1, 1) offset so boundary ancillas at half-integer positions
        (e.g. col −0.5) have visual space in Crumble.
        """
        row, col = self.qubit_coords[qidx]
        return (col + 1, row + 1)

    def offset(self, n: int) -> None:
        """Shift all qubit indices by n. For composing multiple codes in one circuit.

        Warning: resets _circuit and all tracking state. Call before building operations.
        """
        self.qubits = [q + n for q in self.qubits]
        self.measure_qubits = [q + n for q in self.measure_qubits]
        self.z_measure_qubits = [q + n for q in self.z_measure_qubits]
        self.qubit_coords = {k + n: v for k, v in self.qubit_coords.items()}
        self._grid = {k: v + n for k, v in self._grid.items()}
        self.stabilizers = {k + n: [q + n for q in v] for k, v in self.stabilizers.items()}
        self.z_stabilizers = {k + n: [q + n for q in v] for k, v in self.z_stabilizers.items()}
        self._row_start = [s + n for s in self._row_start]
        self._init_circuit()

    def rotate(self) -> None:
        """Rotate the patch 90° clockwise around its center.

        Updates all internal state (coords, indices, stabilizers) and rewrites
        _circuit with remapped qubit indices.
        Local transform: (r, c) → (c, d-1-r).
        """
        if self.d_rows != self.d_cols:
            raise NotImplementedError("rotate() is only supported for square patches.")
        d = self.d
        row_off, col_off = self.origin

        # 1. Compute rotated coords for every qubit
        rotated_coords: dict[int, tuple[float, float]] = {}
        for old_idx, (r, c) in self.qubit_coords.items():
            lr, lc = r - row_off, c - col_off
            new_lr, new_lc = lc, d - 1 - lr
            rotated_coords[old_idx] = (new_lr + row_off, new_lc + col_off)

        # 2. Sort by (new_row, new_col) for canonical index ordering
        sorted_items = sorted(rotated_coords.items(), key=lambda x: (x[1][0], x[1][1]))

        # 3. Build old → new index mapping
        old_to_new = {}
        for new_idx, (old_idx, _) in enumerate(sorted_items):
            old_to_new[old_idx] = new_idx

        # 4. Rebuild internal state
        new_qubit_coords: dict[int, tuple[float, float]] = {}
        new_qubits: list[int] = []
        new_measure_qubits: list[int] = []
        new_z_measure_qubits: list[int] = []
        new_grid: dict[tuple[int, int], int] = {}
        row_starts: dict[int, int] = {}
        old_z_set = set(self.z_measure_qubits)

        for new_idx, (old_idx, (nr, nc)) in enumerate(sorted_items):
            new_qubit_coords[new_idx] = (nr, nc)
            lr, lc = nr - row_off, nc - col_off
            is_integer_row = (lr == int(lr))
            is_integer_col = (lc == int(lc))
            if is_integer_row and is_integer_col:
                new_qubits.append(new_idx)
                new_grid[(int(lr), int(lc))] = new_idx
                r_int = int(lr)
                if r_int not in row_starts:
                    row_starts[r_int] = new_idx
            elif old_idx in old_z_set:
                new_z_measure_qubits.append(new_idx)
            else:
                new_measure_qubits.append(new_idx)

        new_row_start = [row_starts[r] for r in range(d)]

        new_stabilizers = {
            old_to_new[old_mq]: [old_to_new[n] for n in old_neighbors]
            for old_mq, old_neighbors in self.stabilizers.items()
        }
        new_z_stabilizers = {
            old_to_new[k]: [old_to_new[n] for n in old_neighbors]
            for k, old_neighbors in self.z_stabilizers.items()
        }

        # 5. Rewrite _circuit
        new_circuit = stim.Circuit()
        # QUBIT_COORDS first
        for qidx in sorted(new_qubit_coords):
            r, c = new_qubit_coords[qidx]
            new_circuit.append("QUBIT_COORDS", [qidx], [c + 1, r + 1])
        # Remap all other instructions
        for inst in self._circuit:
            if inst.name == "QUBIT_COORDS":
                continue
            new_targets = []
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    new_targets.append(old_to_new[t.value])
                elif t.is_x_target:
                    new_targets.append(stim.target_x(old_to_new[t.value]))
                elif t.is_measurement_record_target:
                    new_targets.append(t)
                else:
                    new_targets.append(t)
            new_circuit.append(inst.name, new_targets, inst.gate_args_copy())

        # 6. Apply updates
        self.qubit_coords = new_qubit_coords
        self.qubits = new_qubits
        self.measure_qubits = new_measure_qubits
        self.z_measure_qubits = new_z_measure_qubits
        self._grid = new_grid
        self._row_start = new_row_start
        self.stabilizers = new_stabilizers
        self.z_stabilizers = new_z_stabilizers
        self._circuit = new_circuit
        # Reset tracking (stale after index remap)
        self._total_measurements = 0
        self._mq_last_rec = {}
        self._measured_pairs = set()
        self._unitary_pairs = set()

    @property
    def n_qubits(self) -> int:
        return self.d_rows * self.d_cols

    @property
    def n_stabs(self) -> int:
        return len(self.measure_qubits)

    def row(self, r: int) -> list[int]:
        """Qubit indices for integer row r."""
        start = self._row_start[r]
        return list(range(start, start + self.d_cols))

    def col(self, c: int) -> list[int]:
        """Qubit indices for column c across all rows."""
        return [self.row(r)[c] for r in range(self.d_rows)]

    def get_observable(self) -> np.ndarray:
        """Logical observable as a binary vector of length d².

        Uses ``self.plus`` to choose the observable:
        - ``plus=True``: X logical (column 0) — measured via MX.
        - ``plus=False``: Z logical (row 0) — measured via M.
        """
        qubit_to_pos = {q: i for i, q in enumerate(self.qubits)}
        obs = np.zeros(self.n_qubits, dtype=np.int8)
        line = self.col(0) if self.plus else self.row(0)
        for q in line:
            obs[qubit_to_pos[q]] = 1
        return obs

    def stab_cx_schedule(self) -> list[list[tuple[int, int]]]:
        """4 CX layers for stabilizer measurement.

        Layer 0: upper-left, Layer 1: upper-right,
        Layer 2: lower-left, Layer 3: lower-right.
        """
        layers: list[list[tuple[int, int]]] = [[], [], [], []]
        for mq, neighbors in self.stabilizers.items():
            row_f, col_f = self.qubit_coords[mq]
            r = int(row_f - self.origin[0])
            c_int = int(col_f - self.origin[1])

            local_row = row_f - self.origin[0]
            if local_row < 0:
                # Top-boundary weight-2 X-mq: place CX in layers 2,3.
                candidates = [
                    None,
                    None,
                    self._grid.get((0, c_int)),
                    self._grid.get((0, c_int + 1)),
                ]
            else:
                candidates = [
                    self._grid.get((r, c_int)),
                    self._grid.get((r, c_int + 1)),
                    self._grid.get((r + 1, c_int)),
                    self._grid.get((r + 1, c_int + 1)),
                ]
            for i, cand in enumerate(candidates):
                if cand is not None and cand in neighbors:
                    layers[i].append((mq, cand))

        return layers

    def z_stab_cx_schedule(self) -> list[list[tuple[int, int]]]:
        """4 CX layers for Z-type stabilizer measurement.

        N-pattern: upper-left, lower-left, upper-right, lower-right.
        CX direction reversed: (dq, zmq) — data as control, ancilla as target.
        """
        # N-pattern: UL=0, LL=1, UR=2, LR=3
        layers: list[list[tuple[int, int]]] = [[], [], [], []]
        row_off, col_off = self.origin
        for zmq, neighbors in self.z_stabilizers.items():
            row_f, col_f = self.qubit_coords[zmq]
            lr = row_f - row_off
            lc = col_f - col_off
            r = int(lr)
            c_int = int(lc) if lc >= 0 else int(lc) - 1  # handle -0.5 → c_int=-1

            # 4 candidate data qubits in Z-pattern order (UL, UR, LL, LR)
            candidates_z = [
                self._grid.get((r, c_int)),
                self._grid.get((r, c_int + 1)),
                self._grid.get((r + 1, c_int)),
                self._grid.get((r + 1, c_int + 1)),
            ]
            # Map Z-pattern → N-pattern: UL→0, LL→1, UR→2, LR→3
            n_pattern = [0, 2, 1, 3]  # Z-index → N-layer
            for z_idx, cand in enumerate(candidates_z):
                if cand is not None and cand in neighbors:
                    n_layer = n_pattern[z_idx]
                    layers[n_layer].append((cand, zmq))  # reversed: (dq, zmq)

        return layers

    # -- Circuit emission helpers --

    @staticmethod
    def _flatten_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """Flatten a list of (control, target) pairs into [c0, t0, c1, t1, ...]."""
        return [q for pair in pairs for q in pair]

    def _emit_cx_layer(self, pairs: list[tuple[int, int]]) -> None:
        """Append CX + TICK for one layer of pairs."""
        flat = self._flatten_pairs(pairs)
        self._circuit.append("CX", flat)
        self._circuit.append("TICK")

    def _emit_reset(self, all_involved: set[int], h_qubits: set[int]) -> None:
        """Append R (non-H qubits) + RX (H qubits) + TICK."""
        sorted_all = sorted(all_involved)
        r_qubits = [q for q in sorted_all if q not in h_qubits]
        rx_qubits = [q for q in sorted_all if q in h_qubits]
        if r_qubits:
            self._circuit.append("R", r_qubits)
        if rx_qubits:
            self._circuit.append("RX", rx_qubits)
        self._circuit.append("TICK")

    # -- Init methods (append to self._circuit) --

    def _init_params(
        self, first_small_left: bool | None = None,
    ) -> tuple[bool, bool, bool]:
        """Derive x_type, rotate, and first_small_left from ``self.plus``.

        Convention: ``first_small_left`` applies to odd row-pair indices;
        even indices get the opposite.  The default is chosen so that the
        outermost pair (d-2, always odd) gets ``first_small_left``.

        Returns:
            (x_type, rotate, first_small_left)
        """
        if self.plus:
            x_type = False
            rotate = True
            if first_small_left is None:
                first_small_left = False
        else:
            x_type = True
            rotate = False
            if first_small_left is None:
                first_small_left = True
        return x_type, rotate, first_small_left

    def unitary_init_row(self, r: int, direction_down: bool = True) -> None:
        """Unitarily initialize a single row pair. Convenience wrapper.

        Args:
            r: Row pair index.
            direction_down: Controls cascade direction.
        """
        self.unitary_init_rows(rows=[r], direction_down=direction_down)

    def _row_pair_gates(
        self,
        r: int,
        direction_down: bool,
        small_left: bool,
        x_type: bool = True,
        rotate: bool = False,
    ) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]]:
        """Compute gates for unitary init of row pair (r, r+1).

        Args:
            r: Row pair index.
            direction_down: Cascade direction.
            small_left: Which end gets the unpaired qubit.
            x_type: If True, cascade CX has H-line as control (X propagation).
                If False, cascade CX has H-line as target (Z propagation).
            rotate: If True, swap rows↔cols (use ``self.row`` instead of
                ``self.col``).

        Returns (h_qubits, pairing_cx, cascade_cx).
        """
        get_line = self.row if rotate else self.col
        # The cascade sweeps over `n_lines` lines, each of length `line_len`.
        # rotate (|+⟩): lines are rows -> n_lines=d_rows, line_len=d_cols.
        # else   (|0⟩): lines are cols -> n_lines=d_cols, line_len=d_rows.
        if rotate:
            n_lines, line_len = self.d_rows, self.d_cols
        else:
            n_lines, line_len = self.d_cols, self.d_rows

        # Map line pair r to line indices.
        # The H-line is the one being initialized; the target receives the cascade.
        h_pair = r if direction_down else r + 1
        t_pair = r + 1 if direction_down else r
        h_line_idx = n_lines - 1 - h_pair
        t_line_idx = n_lines - 1 - t_pair

        h_line = get_line(h_line_idx)
        t_line = get_line(t_line_idx)

        # H targets and pairing CX within the H line.
        # small_left controls the single qubit position.
        if not small_left:
            pair_starts = list(range(1, line_len, 2))
            h_indices = ([0] if x_type else []) + pair_starts
            pairing = [(h_line[i], h_line[i + 1]) for i in pair_starts]
        else:
            pair_starts = list(range(0, line_len - 1, 2))
            h_indices = [i + 1 for i in pair_starts] + ([line_len - 1] if x_type else [])
            pairing = [(h_line[i + 1], h_line[i]) for i in pair_starts]

        h_qubits = [h_line[i] for i in h_indices]

        # Cascade: all line_len positions between H line and target line
        if x_type:
            cascade = [(h_line[i], t_line[i]) for i in range(line_len)]
        else:
            cascade = [(t_line[i], h_line[i]) for i in range(line_len)]

        return h_qubits, pairing, cascade

    def unitary_init_rows(
        self,
        rows: list[int] | None = None,
        direction_down: bool = True,
        first_small_left: bool | None = None,
    ) -> None:
        """Initialize multiple row pairs unitarily. Appends to _circuit.

        Parameters are derived from ``self.plus`` when not specified:
        - ``|0⟩`` (plus=False): first_small_left=True, x_type=True, rotate=False
        - ``|+⟩`` (plus=True): first_small_left=False, x_type=False, rotate=True

        Args:
            rows: Row pair indices. If None, all d-1 pairs in default order.
            direction_down: Controls cascade direction.
            first_small_left: Unpaired qubit side for first row. Alternates.
                Defaults to True for |0⟩, False for |+⟩.
        """
        x_type, rotate, first_small_left = self._init_params(first_small_left)
        n_lines = self.d_rows if rotate else self.d_cols

        if rows is None:
            if direction_down:
                rows = list(range(n_lines - 2, -1, -1))
            else:
                rows = list(range(n_lines - 1))

        for r in rows:
            self._unitary_pairs.add(self._pair_index(r))

        # Collect gates from all row pairs
        all_h: list[int] = []
        all_pairing: list[tuple[int, int]] = []
        all_cascade: list[list[tuple[int, int]]] = []

        for i, r in enumerate(rows):
            small_left = first_small_left if r % 2 == 1 else not first_small_left
            h_qubits, pairing, cascade = self._row_pair_gates(
                r, direction_down, small_left, x_type=x_type, rotate=rotate,
            )
            all_h.extend(h_qubits)
            all_pairing.extend(pairing)
            all_cascade.append(cascade)

        # For |+⟩: each pair's target line needs RX unless the adjacent
        # pair (whose H-line IS this target) is also in the set.
        if rotate:
            row_set = set(rows)
            for r in rows:
                adjacent = r + 1 if direction_down else r - 1
                if adjacent not in row_set:
                    t_pair = r + 1 if direction_down else r
                    t_line_idx = n_lines - 1 - t_pair
                    all_h.extend(self.row(t_line_idx))

        all_involved = {q for cascade in all_cascade for pair in cascade for q in pair}
        all_involved.update(all_h)
        self._emit_reset(all_involved, set(all_h))

        if all_pairing:
            self._emit_cx_layer(all_pairing)

        for cascade in all_cascade:
            self._emit_cx_layer(cascade)

    def unitary_init_bidirectional(self, first_small_left: bool | None = None) -> None:
        """Bidirectional unitary init from middle outward.

        The middle line (col for |0⟩, row for |+⟩) is the shared
        cascade target.  Upper half cascades down toward it, lower half
        cascades up.  The two innermost cascades (both targeting the
        middle) run sequentially; outer steps run in parallel.

        The upper half uses first_small_left, the lower half uses the
        opposite.
        """
        x_type, rotate, _ = self._init_params()
        # Cascade sweeps n_lines lines (rows for |+⟩, cols for |0⟩).
        n_lines = self.d_rows if rotate else self.d_cols

        # Middle line splits the n_lines-1 line pairs evenly.
        mid = (n_lines - 1) // 2  # for n_lines=7: mid=3

        # The correct first_small_left depends on mid parity.
        if first_small_left is None:
            if self.plus:
                first_small_left = mid % 2 == 1
            else:
                first_small_left = mid % 2 == 0

        upper_rows = list(range(mid - 1, -1, -1))       # [2, 1, 0] for n_lines=7
        lower_rows = list(range(mid, n_lines - 1))       # [3, 4, 5] for n_lines=7
        for r in upper_rows + lower_rows:
            self._unitary_pairs.add(self._pair_index(r))

        # Collect all gates from both halves
        all_h: list[int] = []
        all_pairing: list[tuple[int, int]] = []
        steps: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []

        for i, (r_down, r_up) in enumerate(zip(upper_rows, lower_rows)):
            sl_upper = first_small_left if i % 2 == 0 else not first_small_left
            sl_lower = not sl_upper

            h_d, pair_d, casc_d = self._row_pair_gates(
                r_down, direction_down=True, small_left=sl_upper,
                x_type=x_type, rotate=rotate,
            )
            h_u, pair_u, casc_u = self._row_pair_gates(
                r_up, direction_down=False, small_left=sl_lower,
                x_type=x_type, rotate=rotate,
            )

            all_h.extend(h_d + h_u)
            all_pairing.extend(pair_d + pair_u)
            steps.append((casc_d, casc_u))

        # For |+⟩: add RX on the middle line
        if rotate:
            all_h.extend(self.row(mid))

        all_involved = {q for casc_d, casc_u in steps for pair in casc_d + casc_u for q in pair}
        all_involved.update(all_h)
        self._emit_reset(all_involved, set(all_h))

        if all_pairing:
            self._emit_cx_layer(all_pairing)

        for i, (casc_down, casc_up) in enumerate(steps):
            if i == 0:
                # Innermost step: both target the middle line → sequential
                self._emit_cx_layer(casc_down)
                self._emit_cx_layer(casc_up)
            else:
                # Outer steps: disjoint lines → parallel in one TICK
                self._emit_cx_layer(casc_down + casc_up)

    def unitary_init_sequential(self, first_small_left: bool | None = None) -> None:
        """Sequential single-pivot cascade (arXiv:2601.05113, no-ancilla, Fig. 2b/4).

        Like :meth:`unitary_init_rows`, but every CNOT of a plaquette emanates
        from a single pivot, executed sequentially rather than in parallel from
        different controls:

        1. pairing CNOT   — pivot → horizontal partner (within the H-line)
        2. orthogonal CNOT — pivot → the qubit directly across (next line)
        3. diagonal CNOT   — pivot → the partner's across-qubit

        One forward direction (no middle-out), giving depth ``3*(n_lines-1)`` —
        the paper's ``O(d)``, versus the parallel ``unitary_init_rows`` (``O(d)``
        with a smaller constant) and ``unitary_init_bidirectional`` (``O(d/2)``).

        ``|+⟩`` is prepared as the transversal-H dual of the ``|0⟩`` construction
        (reversed CNOTs, swapped reset basis), swept over rows instead of columns.
        """
        # Generate the single-pivot |0>-style circuit (X-fanning, pivot = control)
        # on the appropriate axis: columns for |0>, rows for |+>.
        rotate = self.plus
        n_lines = self.d_rows if rotate else self.d_cols
        _, _, first_small_left = self._init_params(first_small_left)

        rows = list(range(n_lines - 2, -1, -1))
        for r in rows:
            self._unitary_pairs.add(self._pair_index(r))

        pivots: list[int] = []
        # Per line-pair: (pairing, orthogonal, diagonal) CX groups.
        per_pair: list[tuple[list, list, list]] = []
        for r in rows:
            small_left = first_small_left if r % 2 == 1 else not first_small_left
            h_qubits, pairing, cascade = self._row_pair_gates(
                r, direction_down=True, small_left=small_left,
                x_type=True, rotate=rotate,
            )
            pivots.extend(h_qubits)
            # cascade[i] = (h_line[i], t_line[i]); index by H-line position.
            pos_of = {cascade[i][0]: i for i in range(len(cascade))}
            orth: list[tuple[int, int]] = []
            diag: list[tuple[int, int]] = []
            paired: set[int] = set()
            for piv, partner in pairing:
                ip, ipart = pos_of[piv], pos_of[partner]
                orth.append(cascade[ip])                          # pivot -> directly across
                diag.append((cascade[ip][0], cascade[ipart][1]))  # pivot -> partner's across (diagonal)
                paired |= {piv, partner}
            for hq in h_qubits:
                if hq not in paired:
                    orth.append(cascade[pos_of[hq]])              # weight-2 boundary: orthogonal only
            per_pair.append((pairing, orth, diag))

        pivot_set = set(pivots)
        all_data = set(self.qubits)
        if not self.plus:
            # |0>: pivots -> |+> (RX), rest -> |0> (R); emit CNOTs as generated.
            self._emit_reset(all_data, pivot_set)
            transform = lambda layer: layer
        else:
            # |+>: transversal-H dual -> pivots -> |0> (R), rest -> |+> (RX);
            # reverse every CNOT (control <-> target).
            self._emit_reset(all_data, all_data - pivot_set)
            transform = lambda layer: [(b, a) for (a, b) in layer]

        # Each line-pair emits 3 sequential CX layers -> depth 3*(n_lines-1).
        for pairing, orth, diag in per_pair:
            for layer in (pairing, orth, diag):
                if layer:
                    self._emit_cx_layer(transform(layer))

    def init_with_method(self, method: str) -> None:
        """Initialize the patch using a preset method.

        Both methods dispatch to |0⟩ or |+⟩ based on ``self.plus``.

        Methods:
            "full_unitary": One-directional cascade.
            "full_unitary_bidirectional": From middle outward, upper half
                down, lower half up.
            "full_unitary_sequential": One-directional single-pivot cascade
                with sequential/diagonal CNOTs (|0⟩ only).
        """
        if method == "full_unitary":
            self.unitary_init_rows()
        elif method == "full_unitary_bidirectional":
            self.unitary_init_bidirectional()
        elif method == "full_unitary_sequential":
            self.unitary_init_sequential()
        else:
            raise ValueError(f"Unknown init method: {method!r}")

    def _mq_half_row(self, mq: int) -> int:
        """Return the half-row index for a measure qubit.

        Clamps to [0, d-2] so boundary-edge X-mq (at row -0.5 or d-0.5)
        map to the nearest valid half-row.
        """
        row_f = self.qubit_coords[mq][0] - self.origin[0]
        return max(0, min(int(row_f), self.d_rows - 2))

    def _mq_half_col(self, mq: int) -> int:
        """Return the half-column index for a measure qubit.

        Clamps to [0, d-2] so boundary-edge mq (at col -0.5 or d-0.5)
        map to the nearest valid half-column.
        """
        col_f = self.qubit_coords[mq][1] - self.origin[1]
        return max(0, min(int(col_f), self.d_cols - 2))

    def _mq_half_pair(self, mq: int) -> int:
        """Return the half-pair index for a measure qubit, using the correct axis.

        For plus=False (|0⟩): pairs are column-based → uses half-col.
        For plus=True (|+⟩): pairs are row-based → uses half-row.
        """
        if self.plus:
            return self._mq_half_row(mq)
        return self._mq_half_col(mq)

    def _pair_index(self, r: int) -> int:
        """Convert cascade line-pair index r to half-pair index.

        The cascade sweeps over lines: columns for |0⟩ (n_lines = d_cols),
        rows for |+⟩ (n_lines = d_rows).
        """
        n_lines = self.d_rows if self.plus else self.d_cols
        return n_lines - 2 - r

    def _measure_qubits_for_rows(self, rows: list[int] | None, z_type: bool = False) -> list[int]:
        """Return measure qubit indices for the specified row-pair indices (None = all).

        The ``rows`` parameter uses the same indexing as ``unitary_init_rows``:
        row-pair r corresponds to data lines (d-1-r, d-2-r).
        """
        source = self.z_measure_qubits if z_type else self.measure_qubits
        if rows is None:
            return list(source)
        half_indices = {self._pair_index(r) for r in rows}
        return [mq for mq in source if self._mq_half_pair(mq) in half_indices]

    def _cx_schedule_filtered(self, mqs: set[int], z_type: bool = False) -> list[list[tuple[int, int]]]:
        """Return CX schedule filtered to only the given measure qubits.

        For X-type (z_type=False): pairs are (mq, dq), filtered on mq.
        For Z-type (z_type=True): pairs are (dq, zmq), filtered on zmq.
        """
        full = self.z_stab_cx_schedule() if z_type else self.stab_cx_schedule()
        mq_pos = 1 if z_type else 0  # which tuple element is the measure qubit
        return [[pair for pair in layer if pair[mq_pos] in mqs] for layer in full]

    def measure_round(self, rows: list[int] | None = None, z_type: bool | None = None) -> None:
        """Append one stabilizer measurement round to _circuit.

        Args:
            rows: Half-row indices to measure. None = all d-1 half-rows.
            z_type: If True, measure Z-type stabilizers (R/CX(dq→mq)/M).
                    If False, measure X-type (RX/CX(mq→dq)/MX).
                    If None (default), infer from ``self.plus``:
                    plus=True → X-type, plus=False → Z-type.
        """
        if z_type is None:
            z_type = not self.plus
        mqs = self._measure_qubits_for_rows(rows, z_type=z_type)
        mq_set = set(mqs)

        for mq in mqs:
            self._measured_pairs.add(self._mq_half_pair(mq))

        if z_type:
            reset_op, meas_op = "R", "M"
        else:
            reset_op, meas_op = "RX", "MX"
        cx_layers = self._cx_schedule_filtered(mq_set, z_type=z_type)

        # Reset
        self._circuit.append(reset_op, mqs)
        self._circuit.append("TICK")

        # 4 CX layers
        for layer in cx_layers:
            if layer:
                self._emit_cx_layer(layer)

        # Measure
        self._circuit.append(meas_op, mqs)

        # Detectors + record tracking
        end_abs = self._total_measurements + len(mqs)
        for i, mq in enumerate(mqs):
            current_abs = self._total_measurements + i
            if mq in self._mq_last_rec:
                # Round 2+: compare with previous measurement
                prev_abs = self._mq_last_rec[mq]
                vc = self._visual_coords(mq)
                self._circuit.append(
                    "DETECTOR",
                    [stim.target_rec(current_abs - end_abs),
                     stim.target_rec(prev_abs - end_abs)],
                    [vc[0], vc[1], 0],
                )
                self._detector_defs.append([current_abs, prev_abs])
            elif (
                (z_type == (not self.plus) and not self._unitary_pairs)
                or self._mq_half_pair(mq) in self._unitary_pairs
            ):
                vc = self._visual_coords(mq)
                self._circuit.append(
                    "DETECTOR",
                    [stim.target_rec(current_abs - end_abs)],
                    [vc[0], vc[1], 0],
                )
                self._detector_defs.append([current_abs])
            self._mq_last_rec[mq] = current_abs
        self._total_measurements = end_abs

    def final_measurement(self, x_basis: bool | None = None) -> None:
        """Measure all data qubits and add detectors for each stabilizer.

        Args:
            x_basis: Measurement basis and detector type.
                None (default): infer from ``self.plus``.
                True: MX + X-type stabilizers.
                False: M + Z-type stabilizers.

        For unitary-initialized rows: detector = data qubit measurements only (2 or 4).
        For measurement-initialized rows: detector = data qubit measurements +
            last ancilla measurement of that stabilizer (3 or 5).
        """
        self._pre_final_circuit = self._circuit.copy()

        use_x = self.plus if x_basis is None else x_basis
        n_data = len(self.qubits)
        meas_op = "MX" if use_x else "M"

        self._circuit.append(meas_op, self.qubits)

        # Build qubit → position-in-measurement map
        qubit_to_pos = {q: i for i, q in enumerate(self.qubits)}
        end_abs = self._total_measurements + n_data

        skip_row_check = x_basis is not None
        stab_dict = self.stabilizers if use_x else self.z_stabilizers
        for mq, neighbors in stab_dict.items():
            half_pair = self._mq_half_pair(mq)
            if not skip_row_check:
                if half_pair not in self._unitary_pairs and half_pair not in self._measured_pairs:
                    continue
            vc = self._visual_coords(mq)

            rec_targets = []
            abs_indices = []
            for dq in neighbors:
                dq_abs = self._total_measurements + qubit_to_pos[dq]
                rec_targets.append(stim.target_rec(dq_abs - end_abs))
                abs_indices.append(dq_abs)

            # If measurement-initialized, also include last ancilla measurement
            if half_pair in self._measured_pairs and mq in self._mq_last_rec:
                prev_abs = self._mq_last_rec[mq]
                rec_targets.append(stim.target_rec(prev_abs - end_abs))
                abs_indices.append(prev_abs)

            self._circuit.append("DETECTOR", rec_targets, [vc[0], vc[1], 0])
            self._detector_defs.append(abs_indices)

        self._total_measurements = end_abs

    def get_circuit(
        self,
        include_final_measurement: bool = True,
        following_depolarizing_noise: float = 0.0,
    ) -> stim.Circuit:
        """Return a copy of the circuit.

        Args:
            include_final_measurement: If True, include M/MX + detectors.
            following_depolarizing_noise: If > 0, append DEPOLARIZE1 on all
                data qubits after gates, before measurements.
        """
        base = self._pre_final_circuit.copy()
        if following_depolarizing_noise > 0:
            base.append("DEPOLARIZE1", self.qubits, following_depolarizing_noise)
        if include_final_measurement:
            n_pre = len(list(self._pre_final_circuit))
            for i, inst in enumerate(self._circuit):
                if i >= n_pre:
                    base.append(inst)
        return base

    def get_noisy_circuit(
        self,
        noise: NoiseModel,
        include_final_measurement: bool = True,
    ) -> stim.Circuit:
        """Return a copy of the circuit with noise applied.

        Args:
            noise: A noise model (e.g. ``CircuitLevelNoise`` from
                ``fast_surface_code.noise``).
            include_final_measurement: If True, include M/MX + detectors.
        """
        circuit = self.get_circuit(include_final_measurement=include_final_measurement)
        return noise.apply(circuit)

    def get_checks(self) -> np.ndarray:
        """Return the detector parity-check matrix (n_detectors × n_measurements).

        Entry [i, j] = 1 iff detector i depends on measurement j.
        Multiply raw measurement samples by this matrix mod 2 to get syndromes:
            syndromes = (checks @ measurements.T) % 2
        """
        return _build_checks_matrix(self._detector_defs, self._total_measurements)


def _build_checks_matrix(
    detector_defs: list[list[int]], total_measurements: int,
) -> np.ndarray:
    """Build a detector parity-check matrix from detector definitions."""
    n_det = len(detector_defs)
    checks = np.zeros((n_det, total_measurements), dtype=np.int8)
    for i, meas_indices in enumerate(detector_defs):
        for j in meas_indices:
            checks[i, j] = 1
    return checks
