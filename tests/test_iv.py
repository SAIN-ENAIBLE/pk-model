import numpy as np
from scipy.linalg import expm

from pk_model.protocol import Protocol
from pk_model.solve import solve_protocol

# ---------------------------------------------------------------------
# Analytical validation
# Compare numerical ODE solutions against independently known solutions.
# ---------------------------------------------------------------------


def test_central_compartment_analytical_solution():
    """Validate one-compartment IV elimination against its exact solution."""

    central = {"qc": 100.0, "vc": 5.0, "cl": 1.0}
    compartments = {}

    times = np.linspace(0, 10, 50)

    model = Protocol(
        dose=0.0,
        central=central,
        compartments=compartments,
        t_span=(0, 10),
        t_eval=times,
        type="IV",
        Ka=None,
    )

    times = np.linspace(0, 10, 50)
    result = solve_protocol(model, t_span=(0, 10), t_eval=times)

    expected = 100.0 * np.exp(-times / 5.0)

    assert model.solution.success

    np.testing.assert_allclose(
        model.y[0],
        expected,
        rtol=1e-5,
        atol=1e-8,
    )


def test_two_compartment_constant_dose_analytical_solution():
    """Validate two-compartment IV dynamics against a matrix solution."""

    central = {"qc": 100.0, "vc": 5.0, "cl": 1.0}

    compartments = {
        "compartment1": {
            "q1": 0.0,
            "v1": 10.0,
            "Q1": 2.0,
        }
    }

    dose = 10.0

    times = np.linspace(0, 20, 200)

    model = Protocol(
        dose=dose,
        central=central,
        compartments=compartments,
        t_span=(0, 20),
        t_eval=times,
        type="IV",
        Ka=None,
    )

    # Linear system:
    # dy/dt = A @ y + b
    A = np.array(
        [
            [-0.6, 0.2],
            [0.4, -0.2],
        ]
    )

    b = np.array([dose, 0.0])

    y0 = np.array([100.0, 0.0])

    # Augment the matrix so the constant input b can be included
    # in the matrix-exponential reference solution.
    augmented = np.zeros((3, 3))
    augmented[:2, :2] = A
    augmented[:2, 2] = b

    expected = np.array(
        [(expm(augmented * t) @ np.append(y0, 1.0))[:2] for t in times]
    ).T

    assert model.solution.success

    np.testing.assert_allclose(
        model.y,
        expected,
        rtol=1e-6,
        atol=1e-8,
    )


# ---------------------------------------------------------------------
# Clearance behaviour
# Check that elimination behaves consistently with basic PK expectations.
# ---------------------------------------------------------------------


def test_higher_clearance_causes_faster_elimination():
    """Higher clearance should leave less central drug at a fixed time."""

    times = np.linspace(0, 10, 50)

    low_clearance = Protocol(
        dose=0.0,
        central={
            "qc": 100.0,
            "vc": 5.0,
            "cl": 0.5,
        },
        compartments={},
        t_span=(0, 10),
        t_eval=times,
        type="IV",
    )

    high_clearance = Protocol(
        dose=0.0,
        central={
            "qc": 100.0,
            "vc": 5.0,
            "cl": 2.0,
        },
        compartments={},
        t_span=(0, 10),
        t_eval=times,
        type="IV",
    )

    # Greater clearance should remove drug more rapidly.
    assert high_clearance.y[0, -1] < low_clearance.y[0, -1]


def test_no_dose_one_compartment_quantity_decreases():
    """Central amount should monotonically decrease with clearance."""

    times = np.linspace(0, 10, 100)

    model = Protocol(
        dose=0.0,
        central={
            "qc": 100.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={},
        t_span=(0, 10),
        t_eval=times,
        type="IV",
    )

    # Difference between each pair of consecutive time points.
    differences = np.diff(model.y[0])

    # With no drug input, every difference should be negative.
    assert np.all(differences < 0)


# ---------------------------------------------------------------------
# Distribution behaviour
# Check expected movement of drug between central and peripheral states.
# ---------------------------------------------------------------------


def test_drug_distributes_into_empty_peripheral_compartment():
    """Drug should initially move from central into an empty peripheral."""

    times = np.linspace(0, 1, 50)

    model = Protocol(
        dose=0.0,
        central={
            "qc": 100.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={
            "peripheral1": {
                "q1": 0.0,
                "v1": 10.0,
                "Q1": 2.0,
            }
        },
        t_span=(0, 1),
        t_eval=times,
        type="IV",
    )

    # Peripheral compartment starts empty.
    assert model.y[1, 0] == 0.0

    # Positive central-to-peripheral concentration gradient should
    # cause drug to enter the peripheral compartment.
    assert model.y[1, 1] > 0.0


# ---------------------------------------------------------------------
# Mass conservation
# Check that redistribution alone neither creates nor destroys drug.
# ---------------------------------------------------------------------


def test_distribution_conserves_mass_without_clearance():
    """Distribution alone should not create or destroy drug."""

    times = np.linspace(0, 10, 100)

    model = Protocol(
        dose=0.0,
        central={
            "qc": 100.0,
            "vc": 5.0,
            "cl": 0.0,
        },
        compartments={
            "peripheral1": {
                "q1": 0.0,
                "v1": 10.0,
                "Q1": 2.0,
            }
        },
        t_span=(0, 10),
        t_eval=times,
        type="IV",
    )

    # Sum central and peripheral quantities at every time point.
    total_amount = np.sum(
        model.y,
        axis=0,
    )

    # With no dose and no clearance, total drug should remain 100.
    np.testing.assert_allclose(
        total_amount,
        100.0,
        rtol=1e-7,
        atol=1e-8,
    )
