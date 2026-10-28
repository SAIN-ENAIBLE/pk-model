import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot(solution):
    """Plot drug quantity against time for each compartment."""

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, label in enumerate(solution.y):
        label_name = "Central" if i == 0 else f"Peripheral {i}"
        ax.plot(solution.t, label, label=label_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Drug quantity (ng)")
    ax.set_title("Drug quantity over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_solution(sol):
    """Plot one solution's compartment amounts, assuming hours and ng."""
    return plot_solutions([sol])


def plot_solutions(solutions, labels=None):
    """Compare solutions on one graph, assuming all use hours and ng.

    Each solution uses its own time points and may have a different number of
    compartments. Optional labels identify the solutions in the legend.
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

    fig = go.Figure()

    for sol, label in zip(solutions, labels, strict=True):
        for i, amounts in enumerate(sol.y):
            compartment = "Central" if i == 0 else f"Peripheral {i}"
            name = f"{label}: {compartment}" if label else compartment
            fig.add_trace(go.Scatter(x=sol.t, y=amounts, mode="lines", name=name))

    fig.update_layout(
        xaxis_title="Time (hours)",
        yaxis_title="Drug amount (ng)",
        title="Drug Amounts Over Time",
    )

    return fig
