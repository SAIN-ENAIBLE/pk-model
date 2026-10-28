# Imports

import numpy as np

from pk_model.protocol import Protocol

### Tests

# Construct a standard two-compartment IV protocol reused across tests

def make_iv_protocol():
    """Construct a simple two-compartment IV protocol for testing."""
    return Protocol(
        dose=0.0,
        central={
            "qc": 50.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={
            "peripheral1": {
                "q1": 20.0,
                "v1": 10.0,
                "Q1": 2.0,
            }
        },
        t_span=(0.0, 10.0),
        type="IV",
    )

# Check that user inputs are stored and administration type is normalised

def test_protocol_stores_inputs():
    """Protocol should store its supplied configuration."""
    protocol = make_iv_protocol()

    assert protocol.dose == 0.0
    assert protocol.type == "IV"
    assert protocol.compartment_count == 1
    assert protocol.compartment_names == ["peripheral1"]


def test_protocol_type_is_case_insensitive():
    """Protocol should normalise administration type to uppercase."""
    protocol = Protocol(
        dose=0.0,
        central={
            "qc": 50.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={},
        t_span=(0.0, 10.0),
        type="iv",
    )

    assert protocol.type == "IV"

# Peripheral compartment equations
# Check drug transfer between the central and peripheral compartments

def test_compartment_equation():
    """Peripheral exchange should match the PK equation."""
    protocol = make_iv_protocol()

    actual = protocol.compartment_eq(
        Qp=2.0,
        qc=50.0,
        vc=5.0,
        qp=20.0,
        vp=10.0,
    )

    # Central concentration = 50 / 5 = 10
    # Peripheral concentration = 20 / 10 = 2
    # Exchange = 2 * (10 - 2) = 16
    expected = 16.0

    assert actual == expected


def test_all_compartment_equations():
    """Exchange should be calculated independently for each peripheral."""
    protocol = Protocol(
        dose=0.0,
        central={
            "qc": 50.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={
            "peripheral1": {
                "q1": 20.0,
                "v1": 10.0,
                "Q1": 2.0,
            },
            "peripheral2": {
                "q2": 10.0,
                "v2": 20.0,
                "Q2": 0.5,
            },
        },
        t_span=(0.0, 10.0),
        type="IV",
    )

    actual = protocol.all_compartment_eq(
        Qp_list=[2.0, 0.5],
        qc=50.0,
        vc=5.0,
        qp_list=[20.0, 10.0],
        vp_list=[10.0, 20.0],
    )

    # Peripheral 1:
    # 2 * (50/5 - 20/10)
    # = 2 * (10 - 2)
    # = 16
    #
    # Peripheral 2:
    # 0.5 * (50/5 - 10/20)
    # = 0.5 * (10 - 0.5)
    # = 4.75

    expected = [16.0, 4.75]

    np.testing.assert_allclose(actual, expected)

# Central compartment equation
# Check central elimination, drug input, and peripheral distribution

def test_central_equation():
    """Central derivative should include input, elimination and distribution."""
    protocol = make_iv_protocol()

    # central_eq uses the peripheral rates previously calculated.
    protocol.all_compartment_list = [16.0]

    actual = protocol.central_eq(
        absorption=0.0,
        qc=50.0,
        vc=5.0,
        cl=1.0,
    )

    # No input
    # Elimination = (50 / 5) * 1 = 10
    # Peripheral transfer = 16
    #
    # dq_c/dt = 0 - 10 - 16 = -26

    expected = -26.0

    assert actual == expected

# Complete IV ODE system
# Check that central and peripheral equations are assembled correctly

def test_iv_system_equation():
    """IV system equation should return correct central/peripheral rates."""
    protocol = make_iv_protocol()

    actual = protocol.system_eq(
        t=0.0,
        y=[50.0, 20.0],
    )

    expected = [-26.0, 16.0]

    np.testing.assert_allclose(actual, expected)


def test_callable_iv_dose():
    """IV dose may be supplied as a function of time."""

    def dose(t):
        return 2.0 * t

    protocol = Protocol(
        dose=dose,
        central={
            "qc": 100.0,
            "vc": 10.0,
            "cl": 1.0,
        },
        compartments={},
        t_span=(0.0, 10.0),
        type="IV",
    )

    actual = protocol.system_eq(
        t=3.0,
        y=[100.0],
    )

    # Dose at t=3 = 6
    # Elimination = 100 / 10 * 1 = 10
    # Net = 6 - 10 = -4

    expected = [-4.0]

    np.testing.assert_allclose(actual, expected)

# Numerical solver behaviour
# Check the default time grid and dimensions of the integrated solution

def test_default_time_grid_contains_200_points():
    """Protocol should generate 200 evaluation times by default."""
    protocol = make_iv_protocol()

    assert len(protocol.t) == 200


def test_solution_has_one_row_per_compartment():
    """Solution shape should reflect the number of compartments."""
    protocol = make_iv_protocol()

    # Central + one peripheral
    assert protocol.y.shape[0] == 2

# Subcutaneous administration
# Check depot absorption and transfer of absorbed drug into the central state.

def test_sc_system_equation():
    """SC model should include absorption/depot dynamics."""
    protocol = Protocol(
        dose=0.0,
        central={
            "qc": 0.0,
            "vc": 5.0,
            "cl": 1.0,
        },
        compartments={},
        t_span=(0.0, 1.0),
        type="SC",
        Ka=0.5,
    )

    # y = [central quantity, depot quantity]
    actual = protocol.system_eq(
        t=0.0,
        y=[0.0, 100.0],
    )

    # dq0/dt = dose - Ka*q0
    #         = 0 - 0.5*100
    #         = -50
    #
    # absorption into central = Ka*q0 = 50
    #
    # central starts empty, so elimination = 0
    # dqc/dt = 50

    expected = [50.0, -50.0]

    np.testing.assert_allclose(actual, expected)
