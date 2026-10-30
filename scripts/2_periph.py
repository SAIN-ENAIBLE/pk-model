"""IV dosing with two peripherals: uv run python scripts/2_periph.py."""

import matplotlib.pyplot as plt
import numpy as np

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.simulation import Simulation


def main():
    # Illustrative units: ng, hours, L, and L/h.
    # This model has three distribution compartments: central + two peripherals.
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
    clear = Clear(clearance=1.0)
    regime = [Dose.bolus(amount=100.0, t=0.0), Dose.bolus(amount=50.0, t=12.0)]
    times = np.linspace(0.0, 48.0, 481)

    iv = Simulation(model=model, regime=regime, protocol="IV", clear=clear)
    solution = iv.solve(t_end=48.0, t_eval=times)

    # IV rows: central, peripheral 1, peripheral 2.
    print("IV amounts at 48 h:")
    print(f"  Central:      {solution.y[0, -1]:.3f} ng")
    print(f"  Peripheral 1: {solution.y[1, -1]:.3f} ng")
    print(f"  Peripheral 2: {solution.y[2, -1]:.3f} ng")

    _, axes = iv.plot()
    axes.set_title("Central + two peripherals: repeated IV doses")
    plt.show()


if __name__ == "__main__":
    main()
