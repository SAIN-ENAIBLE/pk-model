import matplotlib.pyplot as plt
import numpy as np


def plot(solution):
    """Plot one solution's compartment amounts, assuming hours and ng."""
    return plot_solutions([solution])


def plot_solutions(solutions, labels=None):
    """Compare solutions on one Matplotlib graph, assuming hours and ng.

    Each solution uses its own time points and may have a different number of
    compartments. Optional labels identify the solutions in the legend.
    Retained pre-dose states draw instantaneous bolus jumps at injection times.
    """
    solutions = list(solutions)
    if not solutions:
        raise ValueError("Provide at least one solution to plot.")

    if labels is None:
        if len(solutions) == 1:
            labels = [""]
        else:
            labels = [f"Solution {i + 1}" for i in range(len(solutions))]

    elif isinstance(labels, str) or len(labels) != len(solutions):
        raise ValueError("Provide one label per solution.")

    fig, ax = plt.subplots(figsize=(9, 5))

    for sol, label in zip(solutions, labels, strict=True):
        times, states = sol.t, sol.y
        pre_dose_states = getattr(sol, "pre_dose_states", {})
        if pre_dose_states:
            plot_times, plot_states = [], []
            for t, state in zip(sol.t, sol.y.T, strict=True):
                if t in pre_dose_states:
                    plot_times.append(t)
                    plot_states.append(pre_dose_states[t])
                plot_times.append(t)
                plot_states.append(state)
            times = np.asarray(plot_times)
            states = np.asarray(plot_states).T

        for i, amounts in enumerate(states):
            compartment = "Central" if i == 0 else f"Peripheral {i}"
            if i == getattr(sol, "depot_index", None):
                compartment = "depot"
            name = f"{label}: {compartment}" if label else compartment
            ax.plot(times, amounts, label=name)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Drug amount (ng)")
    ax.set_title("Drug Amounts Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax
