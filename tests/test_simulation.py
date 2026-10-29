import numpy as np
import pytest

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
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


# ---------------------------------------------------------------------
# IV pharmacokinetic behaviour
# Check expected elimination and distribution behaviour.
# ---------------------------------------------------------------------


def test_higher_clearance_causes_faster_elimination():
    """Higher clearance should leave less central drug at a fixed time."""

    times = np.linspace(0, 10, 50)

    low_clearance = Simulation(
        model=make_model(),
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=0.5),
    )

    high_clearance = Simulation(
        model=make_model(),
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=2.0),
    )

    low_solution = low_clearance.solve(
        t_end=10.0,
        t_eval=times,
    )

    high_solution = high_clearance.solve(
        t_end=10.0,
        t_eval=times,
    )

    # Greater clearance should remove drug more rapidly.
    assert high_solution.y[0, -1] < low_solution.y[0, -1]


def test_one_compartment_quantity_decreases_after_iv_bolus():
    """Central amount should monotonically decrease after an IV bolus."""

    times = np.linspace(0, 10, 100)

    simulation = Simulation(
        model=make_model(),
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=1.0),
    )

    solution = simulation.solve(
        t_end=10.0,
        t_eval=times,
    )

    # Difference between each pair of consecutive time points.
    differences = np.diff(solution.y[0])

    # After the initial bolus, with no further drug input,
    # central quantity should continuously decrease.
    assert np.all(differences < 0)


# ---------------------------------------------------------------------
# Distribution behaviour
# Check expected movement of drug between central and peripheral states.
# ---------------------------------------------------------------------


def test_drug_distributes_into_empty_peripheral_compartment():
    """Drug should initially move from central into an empty peripheral."""

    times = np.linspace(0, 1, 50)

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

    simulation = Simulation(
        model=model,
        regime=[Dose.bolus(amount=100.0, t=0.0)],
        protocol="IV",
        clear=Clear(clearance=1.0),
    )

    solution = simulation.solve(
        t_end=1.0,
        t_eval=times,
    )

    # Peripheral compartment starts empty.
    assert solution.y[1, 0] == 0.0

    # The initial central-to-peripheral concentration gradient
    # should cause drug to enter the peripheral compartment.
    assert solution.y[1, 1] > 0.0


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
    assert [line.get_label() for line in axes.lines] == ["Central", "Depot"]


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

# ---------------------------------------------------------------------
# Simulation input validation
# Check that Simulation requires the expected model, regimen and clearance types.
# ---------------------------------------------------------------------


def test_simulation_requires_model():
    """Simulation should require a Model instance."""

    with pytest.raises(TypeError):
        Simulation(
            model="not a model",
            regime=[],
            protocol="IV",
            clear=Clear(1.0),
        )


@pytest.mark.parametrize(
    "regime",
    [
        "not a regimen",
        123,
    ],
)
def test_simulation_requires_dose_sequence(regime):
    """Regimen should be a sequence rather than an arbitrary object."""

    with pytest.raises(TypeError):
        Simulation(
            model=make_model(),
            regime=regime,
            protocol="IV",
            clear=Clear(1.0),
        )


def test_simulation_requires_dose_items():
    """Every regimen item should be a Dose."""

    with pytest.raises(TypeError):
        Simulation(
            model=make_model(),
            regime=["not a dose"],
            protocol="IV",
            clear=Clear(1.0),
        )


def test_simulation_requires_clearance():
    """Simulation should require a Clear instance."""

    with pytest.raises(TypeError):
        Simulation(
            model=make_model(),
            regime=[],
            protocol="IV",
            clear=1.0,
        )