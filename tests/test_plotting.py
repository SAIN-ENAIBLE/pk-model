import matplotlib.pyplot as plt
import numpy as np
import pytest

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.plotting import plot, plot_solutions
from pk_model.simulation import Simulation
from pk_model.solve import SimulationSolution

# ---------------------------------------------------------------------
# Single-solution plotting
# Check compartment data, labels, legends, and axis units are preserved.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("depot_index", [None, 2])
def test_plot_preserves_compartment_data_and_labels(depot_index):
    solution = SimulationSolution(
        t=np.array([0.0, 1.0, 2.0]),
        y=np.array([[100.0, 80.0, 60.0], [0.0, 10.0, 15.0], [20.0, 10.0, 5.0]]),
        depot_index=depot_index,
    )
    labels = [
        "Central",
        "Peripheral 1",
        "depot" if depot_index == 2 else "Peripheral 2",
    ]

    fig, ax = plot(solution)
    try:
        assert fig.axes == [ax]
        assert len(ax.lines) == len(solution.y)
        for line, amounts, label in zip(ax.lines, solution.y, labels, strict=True):
            np.testing.assert_array_equal(line.get_xdata(), solution.t)
            np.testing.assert_array_equal(line.get_ydata(), amounts)
            assert line.get_label() == label
        assert [text.get_text() for text in ax.get_legend().get_texts()] == labels
        assert ax.get_xlabel() == "Time (hours)"
        assert ax.get_ylabel() == "Drug amount (ng)"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------
# Solution comparison
# Check solutions retain their own time grids and compartment labels.
# ---------------------------------------------------------------------


def test_plot_solutions_compares_different_time_grids_and_compartments():
    first = SimulationSolution(t=np.array([0.0, 1.0]), y=np.array([[100.0, 80.0]]))
    second = SimulationSolution(
        t=np.array([0.0, 0.5, 1.0]),
        y=np.array([[0.0, 20.0, 30.0], [100.0, 70.0, 50.0]]),
        depot_index=1,
    )
    fig, ax = plot_solutions([first, second], labels=["IV", "SC"])
    try:
        assert len(ax.lines) == 3
        assert [line.get_label() for line in ax.lines] == [
            "IV: Central",
            "SC: Central",
            "SC: depot",
        ]
        for line, solution, amounts in zip(
            ax.lines, [first, second, second], [first.y[0], *second.y], strict=True
        ):
            np.testing.assert_array_equal(line.get_xdata(), solution.t)
            np.testing.assert_array_equal(line.get_ydata(), amounts)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------
# Bolus jumps with sparse output
# Check IV and SC injections appear as vertical jumps without changing results.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("route", ["IV", "SC"])
@pytest.mark.parametrize("dose_time", [0.0, 2.34567, 10.0])
def test_plot_draws_vertical_bolus_jumps_with_sparse_output(route, dose_time):
    model = Model(CentralParams(5.0, 1.0, (PeripheralParams(10.0, 2.0),)))
    simulation = Simulation(
        model, [Dose.bolus(100.0, dose_time)], route, Clear(1.0), Ka=0.5
    )
    solution = simulation.solve(10.0, t_eval=[0.0, 10.0])
    original_times, original_amounts = solution.t.copy(), solution.y.copy()
    dose_index = solution.depot_index if route == "SC" else 0

    fig, ax = simulation.plot()
    try:
        for i, line in enumerate(ax.lines):
            times, amounts = line.get_xdata(), line.get_ydata()
            jump_indices = np.flatnonzero(times == dose_time)
            assert len(jump_indices) == 2
            before, after = jump_indices
            assert after == before + 1
            # No plotted rise before the injection, including its left limit.
            np.testing.assert_array_equal(amounts[: before + 1], 0.0)
            assert amounts[after] - amounts[before] == pytest.approx(
                100.0 if i == dose_index else 0.0
            )
        # Extra jump vertices belong only to the plot, not the sampled solution.
        np.testing.assert_array_equal(solution.t, original_times)
        np.testing.assert_array_equal(solution.y, original_amounts)
        assert np.all(np.diff(solution.t) > 0)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------
# Independent bolus times
# Check comparison plots show each solution's jumps at its own injection times.
# ---------------------------------------------------------------------


def test_comparison_preserves_independent_bolus_times():
    model = Model(CentralParams(5.0, 1.0))
    iv = Simulation(model, [Dose.bolus(100.0, 2.0)], "IV", Clear(1.0))
    sc = Simulation(model, [Dose.bolus(50.0, 7.0)], "SC", Clear(1.0), Ka=0.5)
    solutions = [sim.solve(10.0, t_eval=[0.0, 10.0]) for sim in [iv, sc]]

    fig, ax = plot_solutions(solutions, labels=["IV", "SC"])
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.0, 2.0, 2.0, 10.0])
        for line in ax.lines[1:]:
            np.testing.assert_array_equal(line.get_xdata(), [0.0, 7.0, 7.0, 10.0])
        np.testing.assert_allclose(ax.lines[0].get_ydata()[:3], [0.0, 0.0, 100.0])
        np.testing.assert_allclose(ax.lines[2].get_ydata()[:3], [0.0, 0.0, 50.0])
    finally:
        plt.close(fig)
