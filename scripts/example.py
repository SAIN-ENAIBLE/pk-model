"""Compare IV and subcutaneous dosing: uv run python scripts/example.py."""

import matplotlib.pyplot as plt
import numpy as np

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.plotting import plot_solutions
from pk_model.simulation import Simulation


def main():
    # Use ng for amounts, hours for time, L for volumes, and L/h for flows.
    # These illustrative parameters describe central and peripheral compartments.
    model = Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(PeripheralParams(volume=10.0, flow=2.0),),
        )
    )
    clear = Clear(clearance=1.0)
    regime = [Dose.bolus(amount=100.0, t=0.0), Dose.bolus(amount=50.0, t=24.0)]

    iv = Simulation(model=model, regime=regime, protocol="IV", clear=clear)
    sc = Simulation(
        model=model,
        regime=regime,
        protocol="SC",
        clear=clear,
        Ka=0.5,  # Absorption rate (1/h); SC doses enter a depot first.
    )

    times = np.linspace(0.0, 48.0, 481)
    iv_solution = iv.solve(t_end=48.0, t_eval=times)
    sc_solution = sc.solve(t_end=48.0, t_eval=times)

    # IV rows: central, peripheral. SC adds a final depot row.
    for route, solution in [("IV", iv_solution), ("SC", sc_solution)]:
        print(
            f"{route}: computed {len(solution.t)} time points; "
            f"central amount at 48 h = {solution.y[0, -1]:.3f} ng"
        )

    _, axes = plot_solutions([iv_solution, sc_solution], labels=["IV", "SC"])
    axes.set_title("IV and subcutaneous dosing: 100 ng at 0 h, 50 ng at 24 h")
    plt.show()


if __name__ == "__main__":
    main()
