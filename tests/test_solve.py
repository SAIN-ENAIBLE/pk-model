# Imports

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.solve import solve_regimen

# Test helpers
# Construct a standard one-compartment model and clearance for reuse


def make_model():
    """Construct a simple one-compartment PK model."""
    return Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
        )
    )


def make_clearance():
    """Construct standard linear clearance."""
    return Clear(clearance=1.0)


# Tests

# ---------------------------------------------------------------------
# Time-span validation
# Check that the simulation interval is finite and has start < end.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "time_span",
    [
        (0.0, 0.0),
        (10.0, 0.0),
        (np.inf, 10.0),
        (0.0, np.inf),
        (np.nan, 10.0),
    ],
)
def test_invalid_time_span_raises_value_error(time_span):
    """Invalid simulation intervals should be rejected."""

    with pytest.raises(ValueError):
        solve_regimen(
            model=make_model(),
            regimen=[],
            clear=make_clearance(),
            t_span=time_span,
        )


# ---------------------------------------------------------------------
# Dose timing validation
# Check that all dose events occur within the simulation interval.
# ---------------------------------------------------------------------


def test_dose_after_time_span_raises_value_error():
    """Dose after the simulation end should be rejected."""

    dose = Dose.bolus(
        amount=100.0,
        t=11.0,
    )

    with pytest.raises(ValueError):
        solve_regimen(
            model=make_model(),
            regimen=[dose],
            clear=make_clearance(),
            t_span=(0.0, 10.0),
        )


# ---------------------------------------------------------------------
# Evaluation-time validation
# Check that requested output times are valid and within t_span.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "t_eval",
    [
        [],
        [[0.0, 1.0]],
        [0.0, 2.0, 1.0],
        [-1.0, 1.0],
        [0.0, 11.0],
        [0.0, np.inf],
        [0.0, np.nan],
    ],
)
def test_invalid_t_eval_raises_value_error(t_eval):
    """Invalid evaluation-time sequences should be rejected."""

    with pytest.raises(ValueError):
        solve_regimen(
            model=make_model(),
            regimen=[],
            clear=make_clearance(),
            t_span=(0.0, 10.0),
            t_eval=t_eval,
        )


# ---------------------------------------------------------------------
# Default output grid
# Check that the solver creates its standard time grid when none is supplied.
# ---------------------------------------------------------------------


def test_default_output_contains_200_time_points():
    """Solver should generate 200 output times when t_eval is omitted."""

    solution = solve_regimen(
        model=make_model(),
        regimen=[],
        clear=make_clearance(),
        t_span=(0.0, 10.0),
    )

    assert len(solution.t) == 200


# ---------------------------------------------------------------------
# IV initial bolus
# Check that an IV bolus at t_start enters the central compartment.
# ---------------------------------------------------------------------


def test_iv_bolus_initialises_central_compartment():
    """IV bolus at t_start should enter the central compartment."""

    solution = solve_regimen(
        model=make_model(),
        regimen=[
            Dose.bolus(
                amount=100.0,
                t=0.0,
            )
        ],
        clear=make_clearance(),
        t_span=(0.0, 1.0),
        t_eval=[0.0, 1.0],
        protocol="IV",
    )

    assert solution.y[0, 0] == pytest.approx(100.0)


# ---------------------------------------------------------------------
# Simultaneous boluses
# Check that multiple doses at the same time are combined.
# ---------------------------------------------------------------------


def test_simultaneous_iv_boluses_are_combined():
    """Boluses at the same time should sum before integration."""

    regimen = [
        Dose.bolus(
            amount=100.0,
            t=0.0,
        ),
        Dose.bolus(
            amount=50.0,
            t=0.0,
        ),
    ]

    solution = solve_regimen(
        model=make_model(),
        regimen=regimen,
        clear=make_clearance(),
        t_span=(0.0, 1.0),
        t_eval=[0.0, 1.0],
        protocol="IV",
    )

    assert solution.y[0, 0] == pytest.approx(150.0)


# ---------------------------------------------------------------------
# Dose-event output
# Check that later dose times are included in the returned solution.
# ---------------------------------------------------------------------


def test_later_bolus_time_is_included_in_output():
    """A dose time should appear in the returned solution times."""

    solution = solve_regimen(
        model=make_model(),
        regimen=[
            Dose.bolus(
                amount=20.0,
                t=5.0,
            )
        ],
        clear=make_clearance(),
        t_span=(0.0, 10.0),
        t_eval=[0.0, 10.0],
        protocol="IV",
    )

    assert 5.0 in solution.t


# ---------------------------------------------------------------------
# SC initial bolus
# Check that an SC bolus enters the depot rather than central.
# ---------------------------------------------------------------------


def test_sc_bolus_initialises_depot():
    """SC bolus at t_start should enter the absorption depot."""

    solution = solve_regimen(
        model=make_model(),
        regimen=[
            Dose.bolus(
                amount=100.0,
                t=0.0,
            )
        ],
        clear=make_clearance(),
        t_span=(0.0, 1.0),
        t_eval=[0.0, 1.0],
        protocol="SC",
        Ka=0.5,
    )

    # Central starts empty.
    assert solution.y[0, 0] == pytest.approx(0.0)

    # Depot contains the SC bolus.
    assert solution.y[1, 0] == pytest.approx(100.0)

    # Solver identifies which row represents the depot.
    assert solution.depot_index == 1


# ---------------------------------------------------------------------
# Solution metadata
# Check route-dependent information attached to SimulationSolution.
# ---------------------------------------------------------------------


def test_iv_solution_has_no_depot_index():
    """IV solution should not identify a depot compartment."""

    solution = solve_regimen(
        model=make_model(),
        regimen=[Dose.bolus(100.0, 0.0)],
        clear=make_clearance(),
        t_span=(0.0, 1.0),
        protocol="IV",
    )

    assert solution.depot_index is None
    assert solution.success


@pytest.mark.parametrize("protocol", ["IV", "SC"])
@pytest.mark.parametrize("has_peripheral", [False, True])
@pytest.mark.parametrize("dense_output_grid", [False, True])
def test_boluses_between_output_times_stop_integration_and_match_exact_solution(
    monkeypatch, protocol, has_peripheral, dense_output_grid
):
    """Doses must be applied exactly, independently of output sampling."""
    segments = []

    def record_segment(**kwargs):
        result = solve_ivp(**kwargs)
        segments.append((kwargs["t_span"], result, result.y.copy()))
        return result

    monkeypatch.setattr("pk_model.solve.solve_ivp", record_segment)
    peripherals = (PeripheralParams(10.0, 2.0),) if has_peripheral else ()
    model = Model(CentralParams(5.0, 1.0, peripherals))
    start, end = 1.0, 10.0
    first, second = 2.34567, 7.12345
    # Unsorted events, simultaneous doses, and doses at both endpoints.
    doses = [
        Dose.bolus(30.0, second),
        Dose.bolus(100.0, start),
        Dose.bolus(20.0, first),
        Dose.bolus(5.0, first),
        Dose.bolus(10.0, end),
    ]
    times = np.linspace(start, end, 101) if dense_output_grid else [start, end]
    solution = solve_regimen(
        model,
        doses,
        make_clearance(),
        (start, end),
        t_eval=times,
        protocol=protocol,
        Ka=0.5 if protocol == "SC" else None,
    )

    # Independent matrix-exponential solution, including every dose exactly once.
    size = model.n_compartments() + (protocol == "SC")
    matrix = np.zeros((size, size))
    matrix[0, 0] = -0.2
    if has_peripheral:
        matrix[0, 0] -= 0.4
        matrix[0, 1] = 0.2
        matrix[1, 0] = 0.4
        matrix[1, 1] = -0.2
    dose_index = size - 1 if protocol == "SC" else 0
    if protocol == "SC":
        matrix[0, -1] = 0.5
        matrix[-1, -1] = -0.5
    expected = np.zeros_like(solution.y)
    for dose in doses:
        jump = np.zeros(size)
        jump[dose_index] = dose.amount
        for i, t in enumerate(solution.t):
            if t >= dose.t:
                expected[:, i] += expm(matrix * (t - dose.t)) @ jump

    np.testing.assert_allclose(solution.y, expected, rtol=1e-7, atol=1e-8)
    assert [span for span, _, _ in segments] == [
        (start, first),
        (first, second),
        (second, end),
    ]
    np.testing.assert_array_equal(solution.t, np.unique([*times, first, second]))
    assert set(solution.pre_dose_states) == {dose.t for dose in doses}
    for t, pre_dose_state in solution.pre_dose_states.items():
        expected_pre = np.zeros(size)
        for dose in doses:
            if dose.t < t:
                jump = np.zeros(size)
                jump[dose_index] = dose.amount
                expected_pre += expm(matrix * (t - dose.t)) @ jump
        np.testing.assert_allclose(pre_dose_state, expected_pre, rtol=1e-7, atol=1e-8)
        expected_jump = np.zeros(size)
        expected_jump[dose_index] = sum(dose.amount for dose in doses if dose.t == t)
        post_dose_state = solution.y[:, np.searchsorted(solution.t, t)]
        np.testing.assert_allclose(
            post_dose_state - pre_dose_state, expected_jump, atol=1e-12
        )
    for (_, boundary), result, original_amounts in segments:
        assert result.t[-1] == boundary
        np.testing.assert_array_equal(result.y, original_amounts)


@pytest.mark.parametrize("bad_result", ["missing_endpoint", "empty", "non_finite"])
def test_invalid_segment_result_is_rejected_before_applying_bolus(
    monkeypatch, bad_result
):
    def invalid_segment(**kwargs):
        result = solve_ivp(**kwargs)
        if bad_result == "missing_endpoint":
            result.t[-1] = np.nextafter(kwargs["t_span"][1], -np.inf)
        elif bad_result == "empty":
            result.t = result.t[:0]
            result.y = result.y[:, :0]
        else:
            result.y[0, -1] = np.nan
        return result

    monkeypatch.setattr("pk_model.solve.solve_ivp", invalid_segment)
    message = "non-finite" if bad_result == "non_finite" else "integration endpoint"
    with pytest.raises(RuntimeError, match=message):
        solve_regimen(
            make_model(), [Dose.bolus(100.0, 2.34567)], make_clearance(), (0.0, 10.0)
        )
