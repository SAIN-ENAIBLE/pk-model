# Imports

import numpy as np
import pytest

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model
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
