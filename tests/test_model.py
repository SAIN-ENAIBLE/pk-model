# For testing model.py in isolation (without route of administration specified)

# Imports

import numpy as np
import pytest

from pk_model.model import CentralParams, Model, PeripheralParams

###
### Input validation
###

# 1 - for peripheral parameters


def test_valid_peripheral_params():
    """Valid peripheral parameters should be stored correctly."""

    peripheral = PeripheralParams(
        volume=10.0,
        flow=2.0,
    )

    assert peripheral.volume == 10.0
    assert peripheral.flow == 2.0


@pytest.mark.parametrize("volume", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_peripheral_volume_raises_value_error(volume):
    """Peripheral volume must be positive and finite."""

    with pytest.raises(ValueError):
        PeripheralParams(
            volume=volume,
            flow=2.0,
        )


@pytest.mark.parametrize("flow", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_peripheral_flow_raises_value_error(flow):
    """Peripheral flow must be positive and finite."""

    with pytest.raises(ValueError):
        PeripheralParams(
            volume=10.0,
            flow=flow,
        )


# 2 - for central parameters


def test_valid_central_params():
    """Valid central parameters should be stored correctly."""

    params = CentralParams(
        central_volume=5.0,
        clearance=1.0,
    )

    assert params.central_volume == 5.0
    assert params.clearance == 1.0
    assert params.peripherals == ()


def test_zero_clearance_disables_elimination():
    model = Model(CentralParams(central_volume=5.0, clearance=0.0))
    assert model.rhs(0.0, [100.0]) == [0.0]


@pytest.mark.parametrize("central_volume", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_central_volume_raises_value_error(central_volume):
    """Central volume must be positive and finite."""

    with pytest.raises(ValueError):
        CentralParams(
            central_volume=central_volume,
            clearance=1.0,
        )


@pytest.mark.parametrize("clearance", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_clearance_raises_value_error(clearance):
    """Central clearance must be positive and finite."""

    with pytest.raises(ValueError):
        CentralParams(
            central_volume=5.0,
            clearance=clearance,
        )


###
### Model construction tests ###
###


def test_model_requires_central_params():
    """Model should only accept a CentralParams object."""

    with pytest.raises(TypeError):
        Model("not CentralParams")


def test_one_compartment_model():
    """No peripheral compartments should yield one total compartment."""
    params = CentralParams(
        central_volume=5.0,
        clearance=1.0,
    )

    model = Model(params)

    assert model.n_compartments() == 1


def test_three_compartment_model():
    """Two peripherals plus central should give three compartments."""
    peripheral_1 = PeripheralParams(
        volume=10.0,
        flow=2.0,
    )

    peripheral_2 = PeripheralParams(
        volume=20.0,
        flow=0.5,
    )

    params = CentralParams(
        central_volume=5.0,
        clearance=1.0,
        peripherals=(peripheral_1, peripheral_2),
    )

    model = Model(params)

    assert model.n_compartments() == 3


###
### model.rhs tests
###


def test_one_compartment_rhs():
    """One-compartment differential should match hand calculation."""
    model = Model(
        CentralParams(
            central_volume=10.0,
            clearance=1.0,
        )
    )

    actual = model.rhs(
        t=0.0,
        y=[100.0],
    )

    # C = q / V = 100 / 10 = 10
    # dq/dt = -CL * C = -1 * 10 = -10

    expected = [-10.0]

    np.testing.assert_allclose(actual, expected)


def test_two_compartment_rhs():
    """Two-compartment derivatives should match hand calculation."""
    peripheral = PeripheralParams(
        volume=10.0,
        flow=2.0,
    )

    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(peripheral,),
        )
    )

    actual = model.rhs(
        t=0.0,
        y=[50.0, 20.0],
    )

    # Central concentration:
    # 50 / 5 = 10
    #
    # Peripheral concentration:
    # 20 / 10 = 2
    #
    # Elimination:
    # 1 * 10 = 10
    #
    # Exchange:
    # 2 * (10 - 2) = 16
    #
    # Central: -10 - 16 = -26
    # Peripheral: +16

    expected = [-26.0, 16.0]

    np.testing.assert_allclose(actual, expected)


def test_three_compartment_rhs():
    """RHS should correctly handle multiple peripheral compartments."""

    peripheral_1 = PeripheralParams(
        volume=10.0,
        flow=2.0,
    )

    peripheral_2 = PeripheralParams(
        volume=20.0,
        flow=0.5,
    )

    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(peripheral_1, peripheral_2),
        )
    )

    actual = model.rhs(
        t=0.0,
        y=[
            50.0,  # Central
            20.0,  # Peripheral-1
            10.0,  # Peripheral-2
        ],
    )

    # Central concentration = 50 / 5 = 10
    #
    # Peripheral-1 concentration = 20 / 10 = 2
    # Exchange 1 = 2 * (10 - 2) = 16
    #
    # Peripheral-2 concentration = 10 / 20 = 0.5
    # Exchange 2 = 0.5 * (10 - 0.5) = 4.75
    #
    # Elimination = 1 * 10 = 10
    #
    # Central derivative = -10 - 16 - 4.75 = -30.75

    expected = [-30.75, 16.0, 4.75]

    np.testing.assert_allclose(actual, expected)


###
### Invalid state test
###


def test_incorrect_state_vector():
    """There must be one state variable per compartment."""

    peripheral = PeripheralParams(
        volume=10.0,
        flow=2.0,
    )

    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1,
            peripherals=(peripheral,),
        )
    )

    # Two-compartment model requires [q_c, q_p1]

    with pytest.raises(ValueError):
        model.rhs(t=0.0, y=[50.0])


@pytest.mark.parametrize(
    "dose_rate",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_dose_rate_raises_value_error(dose_rate):
    """Dose rate must be non-negative and finite."""

    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
        )
    )

    with pytest.raises(ValueError):
        model.rhs(
            t=0.0,
            y=[100.0],
            dose_rate=dose_rate,
        )
