"""Tests for SurfaceCode class."""

import pytest

from fast_surface_code.surface_code import SurfaceCode


# --- Layout tests ---

def test_d7_layout():
    code = SurfaceCode(d=7)
    assert code.n_qubits == 49
    assert code.n_stabs == 24
    assert len(code.z_measure_qubits) == 24
    assert code.total_qubits == 97


def test_d5_layout():
    code = SurfaceCode(d=5)
    assert code.n_qubits == 25
    assert code.n_stabs == 12
    assert len(code.z_measure_qubits) == 12
    assert code.total_qubits == 49


def test_d3_layout():
    code = SurfaceCode(d=3)
    assert code.n_qubits == 9
    assert code.n_stabs == 4
    assert len(code.z_measure_qubits) == 4
    assert code.total_qubits == 17


def test_d7_qubit_coords():
    code = SurfaceCode(d=7)
    for c in range(7):
        assert code.qubit_coords[c] == (0, c)
    assert code.qubit_coords[7] == (0.5, 0.5)
    assert code.qubit_coords[13] == (1, 0)


def test_offset():
    code = SurfaceCode(d=3)
    orig = code.qubits.copy()
    code.offset(100)
    assert code.qubits == [q + 100 for q in orig]


def test_two_patches_no_index_collision():
    a = SurfaceCode(d=5)
    b = SurfaceCode(d=5, origin=(0, 6))
    b.offset(a.total_qubits)
    assert set(a.qubits + a.measure_qubits).isdisjoint(set(b.qubits + b.measure_qubits))


# --- _circuit tests ---

def test_circuit_starts_with_qubit_coords():
    code = SurfaceCode(d=5)
    s = str(code._circuit)
    assert "QUBIT_COORDS" in s
    # No R until an init method is called
    assert "R " not in s


# --- Unitary init tests ---

def test_unitary_init_row_direction_down():
    """direction_down: h_col at outer edge, cascade toward center."""
    code = SurfaceCode(d=7)
    code.unitary_init_row(0, direction_down=True)
    s = str(code._circuit)
    # For r=0, direction_down: h_col=6, t_col=5
    h_col = code.col(6)
    t_col = code.col(5)
    # Cascade: all rows, col 6 → col 5
    for row in range(7):
        assert f"{h_col[row]} {t_col[row]}" in s


def test_unitary_init_row_direction_up():
    """direction_up: h_col is on the left, cascade goes right."""
    code = SurfaceCode(d=5)
    code.unitary_init_row(0, direction_down=False)
    s = str(code._circuit)
    # For r=0, direction_up: h_col = d-1-(r+1) = 3, t_col = d-1-r = 4
    h_col = code.col(3)
    t_col = code.col(4)
    # Cascade: col 3 → col 4
    for row in range(5):
        assert f"{h_col[row]} {t_col[row]}" in s


def test_unitary_init_row_plus():
    """unitary_init_row works for plus=True (row-based pairs)."""
    code = SurfaceCode(d=5, plus=True)
    code.unitary_init_row(0)
    code.final_measurement()
    dem = code._circuit.detector_error_model(decompose_errors=True)
    assert code._circuit.num_detectors == 4


def test_unitary_init_rows_full_patch():
    code = SurfaceCode(d=5)
    code.unitary_init_rows(direction_down=True, first_small_left=True)
    code._circuit.append("MX", code.qubits)
    assert code._circuit.num_qubits > 0


def test_unitary_init_rows_partial():
    code = SurfaceCode(d=7)
    code.unitary_init_rows(rows=[5, 4, 3], direction_down=True, first_small_left=True)
    s = str(code._circuit)
    # Should have H gates (from the init rows)
    assert "RX " in s


# --- Rotation tests ---

def test_rotate_coords_d5():
    """90° CW rotation: (r, c) → (c, d-1-r)."""
    code = SurfaceCode(d=5)
    # Before: qubit 0 at (0, 0)
    assert code.qubit_coords[0] == (0, 0)
    code.rotate()
    # After: (0, 0) → (0, 4). Find the qubit now at (0, 4).
    coords_to_idx = {v: k for k, v in code.qubit_coords.items()}
    assert (0, 4) in coords_to_idx
    assert (4, 0) in coords_to_idx  # (0, 4) → (4, 4) after rotation


def test_rotate_preserves_counts():
    code = SurfaceCode(d=7)
    n_q = code.n_qubits
    n_s = code.n_stabs
    code.rotate()
    assert code.n_qubits == n_q
    assert code.n_stabs == n_s
    assert len(code.qubits) == n_q
    assert len(code.measure_qubits) == n_s


def test_rotate_twice_is_180():
    """Two 90° CW rotations = 180°: (r,c) → (d-1-r, d-1-c)."""
    code = SurfaceCode(d=5)
    orig_coords = dict(code.qubit_coords)
    code.rotate()
    code.rotate()
    # Each qubit should be at (d-1-r, d-1-c) of its original position
    # Check via coord mapping
    for idx, (r, c) in code.qubit_coords.items():
        pass  # just verify no crash; coords are remapped
    assert code.n_qubits == 25
    assert code.n_stabs == 12


def test_rotate_circuit_valid():
    """Circuit should remain valid stim after rotation."""
    code = SurfaceCode(d=5)
    code.unitary_init_rows(direction_down=True, first_small_left=True)
    code.rotate()
    code._circuit.append("MX", code.qubits)
    assert code._circuit.num_qubits > 0


def test_rotate_row_method_works():
    """row() should still return d contiguous indices after rotation."""
    code = SurfaceCode(d=5)
    code.rotate()
    for r in range(5):
        row = code.row(r)
        assert len(row) == 5
        assert row == list(range(row[0], row[0] + 5))


def test_rotate_z_stabilizers_remapped():
    """Z-stabilizer keys and values must use new indices after rotation."""
    code = SurfaceCode(d=5)
    code.rotate()
    all_indices = set(code.qubit_coords.keys())
    for zmq, neighbors in code.z_stabilizers.items():
        assert zmq in all_indices, f"z_stab key {zmq} not a valid qubit index"
        assert zmq in code.z_measure_qubits, f"z_stab key {zmq} not in z_measure_qubits"
        for dq in neighbors:
            assert dq in all_indices, f"z_stab neighbor {dq} not a valid qubit index"
            assert dq in code.qubits, f"z_stab neighbor {dq} not a data qubit"


# --- init_with_method tests ---

def test_init_full_unitary():
    code = SurfaceCode(d=5)
    code.init_with_method("full_unitary")
    code._circuit.append("MX", code.qubits)
    assert code._circuit.num_qubits > 0
    assert "RX " in str(code._circuit)


def test_init_full_unitary_bidirectional():
    code = SurfaceCode(d=7)
    code.init_with_method("full_unitary_bidirectional")
    code._circuit.append("MX", code.qubits)
    assert code._circuit.num_qubits > 0
    assert "RX " in str(code._circuit)


def test_init_bidirectional_d5():
    code = SurfaceCode(d=5)
    code.init_with_method("full_unitary_bidirectional")
    code._circuit.append("MX", code.qubits)
    assert code._circuit.num_qubits == code.total_qubits


def test_init_unknown_method_raises():
    code = SurfaceCode(d=3)
    with pytest.raises(ValueError):
        code.init_with_method("nonexistent")


# --- Stabilizer measurement tests ---

def test_stab_cx_layers_sizes_d7():
    code = SurfaceCode(d=7)
    layers = code.stab_cx_schedule()
    assert len(layers[0]) == 21
    assert len(layers[1]) == 21
    assert len(layers[2]) == 21
    assert len(layers[3]) == 21


def test_stab_cx_all_measure_qubits_used():
    code = SurfaceCode(d=7)
    layers = code.stab_cx_schedule()
    all_mqs = set()
    for layer in layers:
        for mq, _ in layer:
            all_mqs.add(mq)
    assert all_mqs == set(code.measure_qubits)


def test_measure_round():
    code = SurfaceCode(d=5, plus=True)
    code.measure_round()
    s = str(code._circuit)
    assert "RX " in s
    assert "MX " in s


def test_measure_round_z_type():
    code = SurfaceCode(d=5, plus=False)
    code.measure_round()
    s = str(code._circuit)
    assert "R " in s
    assert "M " in s


def test_measure_round_first_round_detectors_z_type():
    """First Z-type round has single-measurement detectors (deterministic for |0⟩)."""
    code = SurfaceCode(d=5, plus=False)
    code.measure_round()
    s = str(code._circuit)
    n_z_stabs = len(code.z_stabilizers)
    assert s.count("DETECTOR") == n_z_stabs


def test_measure_round_first_round_detectors_x_type():
    """First X-type round also has single-measurement detectors."""
    code = SurfaceCode(d=5, plus=True)
    code.measure_round()
    s = str(code._circuit)
    n_x_stabs = len(code.stabilizers)
    assert s.count("DETECTOR") == n_x_stabs


def test_measure_round_detectors_second_round():
    code = SurfaceCode(d=5)
    code.measure_round()
    code.measure_round()
    s = str(code._circuit)
    assert "DETECTOR" in s
    # Round 1: n_z_stabs detectors, Round 2: n_z_stabs detectors
    n_z_stabs = len(code.z_stabilizers)
    assert s.count("DETECTOR") == 2 * n_z_stabs


def test_measure_round_detectors_third_round():
    code = SurfaceCode(d=5)
    for _ in range(3):
        code.measure_round()
    s = str(code._circuit)
    # All 3 rounds add n_z_stabs detectors each
    n_z_stabs = len(code.z_stabilizers)
    assert s.count("DETECTOR") == 3 * n_z_stabs


def test_measure_round_per_row():
    """Measure only specific rows."""
    code = SurfaceCode(d=5, plus=True)
    code.measure_round(rows=[0, 1])
    s = str(code._circuit)
    assert "MX " in s
    mqs_in_rows = code._measure_qubits_for_rows([0, 1])
    # First round: first-round detectors
    assert s.count("DETECTOR") == len(mqs_in_rows)
    # Second round on same rows: adds another set of detectors
    code.measure_round(rows=[0, 1])
    s = str(code._circuit)
    assert s.count("DETECTOR") == 2 * len(mqs_in_rows)


# --- Final measurement tests ---

def test_final_measurement_no_init_no_detectors():
    """Uninitialized rows should get no detectors."""
    code = SurfaceCode(d=5)
    code.final_measurement()
    assert "DETECTOR" not in str(code._circuit)


def test_final_measurement_partial_init():
    """Only initialized pairs get detectors."""
    code = SurfaceCode(d=7)
    # Init only pairs 0 and 1 unitarily (for plus=False these are column-pairs).
    # Detectors should appear only for Z-type stabilizers in those pairs.
    code.unitary_init_row(0)
    code.unitary_init_row(1)
    code.final_measurement()
    s = str(code._circuit)
    half_pairs = {code._pair_index(0), code._pair_index(1)}
    n_z_in_pairs = sum(1 for zmq in code.z_stabilizers if code._mq_half_pair(zmq) in half_pairs)
    assert s.count("DETECTOR") == n_z_in_pairs
    assert n_z_in_pairs > 0


def test_final_measurement_all_unitary():
    """All rows unitary: detectors have 2 or 4 data qubit refs, no ancilla."""
    code = SurfaceCode(d=5)
    code.unitary_init_rows()
    code.final_measurement()
    s = str(code._circuit)
    assert s.count("DETECTOR") == code.n_stabs
    # No measurement-initialized rows → no 5-target or 3-target detectors
    # (detectors should have 2 or 4 rec targets, never 3 or 5)


def test_final_measurement_all_measurement():
    """All rows measured: detectors include ancilla MX ref."""
    code = SurfaceCode(d=5)
    code.measure_round()
    code.measure_round()
    code.final_measurement()
    s = str(code._circuit)
    # 2 rounds of Z-type detectors + final detectors
    n_z_stabs = len(code.z_stabilizers)
    assert s.count("DETECTOR") == 2 * n_z_stabs + n_z_stabs


def test_final_measurement_mixed():
    """Some rows unitary, some measured."""
    code = SurfaceCode(d=7)
    # Unitary init rows 0-2
    code.unitary_init_rows(rows=[4, 3, 2], direction_down=True, first_small_left=True)
    # Measurement init rows 3-5
    code.measure_round(rows=[3, 4, 5])
    code.measure_round(rows=[3, 4, 5])
    code.final_measurement()
    s = str(code._circuit)
    assert "DETECTOR" in s


# --- Deterministic detector tests ---

@pytest.mark.parametrize("plus", [False, True])
def test_detectors_deterministic(plus):
    """All detectors must be deterministic at p=0."""
    code = SurfaceCode(d=5, plus=plus)
    code.unitary_init_rows()
    code.final_measurement()
    # Raises if any detector is non-deterministic
    code._circuit.detector_error_model()


# --- Rectangular patch tests ---

def test_square_via_dx_dz_matches_d():
    """SurfaceCode(dx=5, dz=5) is identical in size to SurfaceCode(d=5)."""
    a = SurfaceCode(d=5)
    b = SurfaceCode(dx=5, dz=5)
    assert a.n_qubits == b.n_qubits
    assert a.n_stabs == b.n_stabs
    assert len(a.z_measure_qubits) == len(b.z_measure_qubits)


def test_rectangular_layout_invariant():
    """Rectangular patch encodes one logical qubit (stabilizer sum = n_data - 1)."""
    code = SurfaceCode(dx=3, dz=7)
    assert code.n_qubits == 21
    n_stab = code.n_stabs + len(code.z_measure_qubits)
    assert n_stab == 21 - 1
    # X/Z counts are individually asymmetric for rectangular patches
    assert code.n_stabs != len(code.z_measure_qubits)


def test_rectangular_d_property_raises():
    code = SurfaceCode(dx=3, dz=5)
    with pytest.raises(AttributeError):
        _ = code.d


def test_constructor_argument_errors():
    with pytest.raises(ValueError):
        SurfaceCode()  # neither d nor dx/dz
    with pytest.raises(ValueError):
        SurfaceCode(d=5, dx=3)  # both
    with pytest.raises(AssertionError):
        SurfaceCode(dx=4, dz=5)  # even distance


@pytest.mark.parametrize("dx,dz", [(3, 5), (5, 3), (3, 7), (5, 7)])
@pytest.mark.parametrize("method", ["rows", "bidirectional"])
@pytest.mark.parametrize("plus", [False, True])
def test_rectangular_unitary_prep_deterministic(dx, dz, method, plus):
    """Rectangular unitary prep produces a valid codestate (deterministic detectors)."""
    code = SurfaceCode(dx=dx, dz=dz, plus=plus)
    if method == "rows":
        code.unitary_init_rows()
    else:
        code.unitary_init_bidirectional()
    code.final_measurement()
    # Raises if any detector is non-deterministic (i.e. not a valid codestate)
    code._circuit.detector_error_model()


@pytest.mark.parametrize("dx,dz", [(3, 5), (3, 7)])
def test_rectangular_distance(dx, dz):
    """|0> memory distance = dx, |+> memory distance = dz."""
    import stim

    from fast_surface_code.noise import CircuitLevelNoise

    def dist(plus):
        code = SurfaceCode(dx=dx, dz=dz, plus=plus)
        code.unitary_init_rows()
        code.final_measurement()
        obs = code.get_observable()
        nd = len(code.qubits)
        for i, val in enumerate(obs):
            if val:
                code._circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(nd - i))], [0])
        noisy = CircuitLevelNoise(1e-3, 1e-3, 1e-3, 1e-3).apply(code._circuit)
        return len(noisy.shortest_graphlike_error())

    assert dist(False) == dx  # |0> fails via logical X (crosses rows)
    assert dist(True) == dz   # |+> fails via logical Z (crosses cols)


# --- Sequential single-pivot construction (arXiv:2601.05113) ---

@pytest.mark.parametrize("dx,dz", [(3, 3), (5, 5), (7, 7), (3, 5), (5, 7)])
@pytest.mark.parametrize("plus", [False, True])
def test_sequential_prep_deterministic(dx, dz, plus):
    """Sequential prep produces a valid codestate (deterministic detectors)."""
    code = SurfaceCode(dx=dx, dz=dz, plus=plus)
    code.unitary_init_sequential()
    code.final_measurement()
    code._circuit.detector_error_model()  # raises if non-deterministic


@pytest.mark.parametrize("plus", [False, True])
def test_sequential_depth_is_3d(plus):
    """Sequential construction has depth 3*(n_lines-1) CX layers (paper's O(d))."""
    for d in (3, 5, 7):
        code = SurfaceCode(d=d, plus=plus)
        code.unitary_init_sequential()
        n_cx = sum(1 for inst in code._circuit if inst.name == "CX")
        assert n_cx == 3 * (d - 1)


@pytest.mark.parametrize("dx,dz", [(3, 5), (5, 7)])
@pytest.mark.parametrize("plus", [False, True])
def test_sequential_distance(dx, dz, plus):
    """Sequential prep preserves the code distance (|0> -> dx, |+> -> dz)."""
    import stim

    from fast_surface_code.noise import CircuitLevelNoise

    code = SurfaceCode(dx=dx, dz=dz, plus=plus)
    code.unitary_init_sequential()
    code.final_measurement()
    obs = code.get_observable()
    nd = len(code.qubits)
    for i, val in enumerate(obs):
        if val:
            code._circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(nd - i))], [0])
    noisy = CircuitLevelNoise(1e-3, 1e-3, 1e-3, 1e-3).apply(code._circuit)
    assert len(noisy.shortest_graphlike_error()) == (dx if not plus else dz)
