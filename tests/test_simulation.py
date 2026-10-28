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
    figure = simulation.plot()
    assert len(figure.data) == 1
