import numpy as np
import pytest

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model
from pk_model.simulation import Simulation


def make_model():
    return Model(CentralParams(central_volume=5.0, clearance=1.0))


def test_simulation_applies_bolus_and_uses_linear_clearance():
    simulation = Simulation(
        model=make_model(),
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=1.0),
    )

    solution = simulation.solve(t_end=10.0, t_eval=[0.0, 5.0, 10.0])

    expected = 100.0 * np.exp(-np.array([0.0, 5.0, 10.0]) / 5.0)
    np.testing.assert_allclose(solution.y[0], expected, rtol=1e-7)


def test_simulation_applies_later_bolus_after_prior_dynamics():
    simulation = Simulation(
        model=make_model(),
        regime=[
            Dose.bolus(amount=100.0, t=0.0),
            Dose.bolus(amount=20.0, t=5.0),
        ],
        protocol="IV",
        clear=Clear(clearance=1.0),
    )

    solution = simulation.solve(t_end=10.0, t_eval=[0.0, 5.0, 10.0])

    amount_at_five = 100.0 * np.exp(-1.0) + 20.0
    expected_at_ten = amount_at_five * np.exp(-1.0)
    np.testing.assert_allclose(
        solution.y[0, 1:], [amount_at_five, expected_at_ten], rtol=1e-7
    )


def test_plot_requires_solution_and_returns_figure_after_solve():
    simulation = Simulation(
        model=make_model(),
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=1.0),
    )

    with pytest.raises(RuntimeError, match="Call solve\\(\\) before plot\\(\\)"):
        simulation.plot()

    simulation.solve(t_end=1.0)
    figure, axes = simulation.plot()
    assert figure.axes == [axes]
    assert len(axes.lines) == 1


@pytest.mark.parametrize("Ka", [0.2, 0.5])
def test_sc_repeated_boluses_match_analytical_absorption(Ka):
    doses = [Dose.bolus(100.0, 0.0), Dose.bolus(20.0, 5.0)]
    simulation = Simulation(make_model(), doses, "subcutaneous", Clear(1.0), Ka=Ka)
    solution = simulation.solve(10.0, t_eval=[0.0, 2.0, 5.0, 10.0])

    central = np.zeros_like(solution.t)
    depot = np.zeros_like(solution.t)
    for dose in doses:
        elapsed = solution.t - dose.t
        active = elapsed >= 0
        dt = elapsed[active]
        depot[active] += dose.amount * np.exp(-Ka * dt)
        if Ka == 0.2:
            central[active] += dose.amount * Ka * dt * np.exp(-Ka * dt)
        else:
            central[active] += (
                dose.amount * Ka / (Ka - 0.2) * (np.exp(-0.2 * dt) - np.exp(-Ka * dt))
            )

    np.testing.assert_allclose(solution.y, [central, depot], rtol=1e-7, atol=1e-9)
    assert solution.depot_index == 1
    _, axes = simulation.plot()
    assert [line.get_label() for line in axes.lines] == ["Central", "depot"]


@pytest.mark.parametrize("route", ["IM", "ID", "intramuscular", "intradermal"])
def test_unimplemented_routes(route):
    with pytest.raises(NotImplementedError):
        Simulation(make_model(), [], route, Clear(1.0))


@pytest.mark.parametrize("Ka", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_sc_requires_positive_finite_absorption_rate(Ka):
    with pytest.raises(ValueError, match="Ka"):
        Simulation(make_model(), [], "SC", Clear(1.0), Ka=Ka)


def test_unknown_route_is_rejected():
    with pytest.raises(ValueError, match="Unknown administration route"):
        Simulation(make_model(), [], "unknown", Clear(1.0))
