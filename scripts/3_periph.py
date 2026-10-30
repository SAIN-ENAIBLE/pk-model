"""Compare IV models with two and three peripherals: uv run python scripts/3_periph.py."""

import matplotlib.pyplot as plt
import numpy as np

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.plotting import plot_solutions
from pk_model.simulation import Simulation


def main():
    # Illustrative units: ng, hours, L, and L/h.
    peripherals = (
        PeripheralParams(volume=10.0, flow=2.0),
        PeripheralParams(volume=20.0, flow=0.5),
        PeripheralParams(volume=30.0, flow=0.15),
    )
    two_peripheral_model = Model(
        CentralParams(central_volume=5.0, clearance=1.0, peripherals=peripherals[:2])
    )
    three_peripheral_model = Model(
        CentralParams(central_volume=5.0, clearance=1.0, peripherals=peripherals)
    )

    # Reuse exactly the same dosing schedule, IV route, and clearance
    # with both models to isolate the effect of adding a third peripheral.
    clear = Clear(clearance=1.0)
    regime = [Dose.bolus(amount=100.0, t=float(t)) for t in [0, 24, 48]]
    times = np.linspace(0.0, 72.0, 721)
    labels = ["Two peripherals", "Three peripherals"]
    solutions = []

    for label, model in zip(
        labels, [two_peripheral_model, three_peripheral_model], strict=True
    ):
        simulation = Simulation(model=model, regime=regime, protocol="IV", clear=clear)
        solution = simulation.solve(t_end=72.0, t_eval=times)
        solutions.append(solution)
        print(f"{label}: central amount at 72 h = {solution.y[0, -1]:.3f} ng")

    # The larger IV model has four rows:
    # central, peripheral 1, peripheral 2, peripheral 3.
    print(f"Peripheral 3 amount at 72 h = {solutions[1].y[3, -1]:.3f} ng")
    figure, axes = plot_solutions(solutions, labels=labels)
    axes.set_title("Same IV regimen: central + two or three peripherals")
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
