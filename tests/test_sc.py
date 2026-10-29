"""Validate the depot, central, and peripheral SC equations."""

import numpy as np
import pytest
from scipy.linalg import expm

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.simulation import Simulation


def make_model():
    return Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(PeripheralParams(volume=10.0, flow=2.0),),
        )
    )


@pytest.mark.parametrize("dose_rate", [10.0, lambda t: 2.0 * t])
def test_sc_equations_and_mass_balance(dose_rate):
    # At t=5, Dose(t)=10, absorption=0.5*100=50,
    # elimination=1*50/5=10, exchange=2*(50/5-20/10)=16.
    rates = make_model().rhs(5.0, [50.0, 20.0, 100.0], dose_rate, protocol="SC", Ka=0.5)
    np.testing.assert_allclose(rates, [24.0, 16.0, -40.0])
    # Internal transfers cancel: total derivative = input - elimination.
    assert sum(rates) == 10.0 - 10.0


def test_sc_multiple_peripherals_and_clearance_override():
    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(
                PeripheralParams(volume=10.0, flow=2.0),
                PeripheralParams(volume=20.0, flow=0.5),
            ),
        )
    )
    rates = model.rhs(
        0.0,
        [50.0, 20.0, 10.0, 100.0],
        Ka=0.5,
        dose_rate=10.0,
        clear=Clear(2.0),
        protocol="SC",
    )
    np.testing.assert_allclose(rates, [9.25, 16.0, 4.75, -40.0])
    assert sum(rates) == -10.0


@pytest.mark.parametrize("input_kind", ["none", "constant", "ramp"])
def test_sc_solution_matches_matrix_exponential_with_repeated_doses(input_kind):
    # Independent linear system in order [central, peripheral, depot, time, 1].
    matrix = np.zeros((5, 5))
    matrix[:3, :3] = [[-0.6, 0.2, 0.5], [0.4, -0.2, 0.0], [0.0, 0.0, -0.5]]
    matrix[3, 4] = 1.0
    if input_kind == "constant":
        matrix[2, 4] = 3.0
        dose_rate = 3.0
    elif input_kind == "ramp":
        matrix[2, 3] = 2.0
        dose_rate = lambda t: 2.0 * t
    else:
        dose_rate = 0.0

    doses = [Dose.bolus(100.0, 0.0), Dose.bolus(25.0, 5.0)]
    simulation = Simulation(make_model(), doses, "SC", Clear(1.0), Ka=0.5)
    solution = simulation.solve(
        10.0, t_eval=np.linspace(0, 10, 41), dose_rate=dose_rate
    )

    expected = []
    for t in solution.t:
        state = (expm(matrix * t) @ [0.0, 0.0, 0.0, 0.0, 1.0])[:3]
        for dose in doses:
            if t >= dose.t:
                state += (expm(matrix * (t - dose.t)) @ [0, 0, dose.amount, 0, 0])[:3]
        expected.append(state)

    np.testing.assert_allclose(solution.y, np.array(expected).T, rtol=1e-7, atol=1e-8)
    assert solution.depot_index == 2
    np.testing.assert_allclose(solution.y[:, 0], [0.0, 0.0, 100.0])


@pytest.mark.parametrize("rate", [-1.0, np.inf, np.nan, lambda t: -1.0])
def test_invalid_sc_input_rate(rate):
    simulation = Simulation(make_model(), [], "SC", Clear(1.0), Ka=0.5)
    with pytest.raises(ValueError, match="Dose rate"):
        simulation.solve(1.0, dose_rate=rate)


@pytest.mark.parametrize("Ka", [None, 0.0, -1.0, np.inf, np.nan])
def test_invalid_sc_absorption_rate(Ka):
    with pytest.raises(ValueError, match="Ka"):
        make_model().rhs(0.0, [0.0, 0.0, 100.0], protocol="SC", Ka=Ka)


def test_sc_requires_depot_state():
    with pytest.raises(ValueError, match="SC requires 3 states"):
        make_model().rhs(0.0, [0.0, 100.0], protocol="SC", Ka=0.5)


def test_sc_internal_transfer_allows_negative_solver_states():
    rates = make_model().rhs(0.0, [0.0, 0.0, -1e-12], protocol="SC", Ka=0.5)
    np.testing.assert_allclose(rates, [-5e-13, 0.0, 5e-13], atol=1e-20)


@pytest.mark.parametrize("route", ["SC", "sc", "subcutaneous"])
def test_rhs_accepts_sc_route_names(route):
    rates = make_model().rhs(0.0, [0.0, 0.0, 100.0], protocol=route, Ka=0.5)
    np.testing.assert_allclose(rates, [50.0, 0.0, -50.0])


@pytest.mark.parametrize("route", ["IV", "iv", "intravenous"])
def test_rhs_accepts_iv_route_names(route):
    rates = make_model().rhs(0.0, [50.0, 20.0], protocol=route)
    np.testing.assert_allclose(rates, [-26.0, 16.0])


@pytest.mark.parametrize("route", ["IM", "ID", "intramuscular", "intradermal"])
def test_rhs_rejects_unimplemented_routes(route):
    with pytest.raises(NotImplementedError):
        make_model().rhs(0.0, [0.0, 0.0], protocol=route)


def test_rhs_rejects_unknown_route():
    with pytest.raises(ValueError, match="Unknown administration route"):
        make_model().rhs(0.0, [0.0, 0.0], protocol="unknown")


def test_rhs_rejects_non_string_route():
    with pytest.raises(TypeError, match="route string"):
        make_model().rhs(0.0, [0.0, 0.0], protocol=None)
