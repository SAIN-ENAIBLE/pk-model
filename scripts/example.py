import matplotlib.pyplot as plt

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model
from pk_model.simulation import Simulation

model = Model(CentralParams(central_volume=5.0, clearance=1.0))
regime = [Dose.bolus(amount=100.0, t=0.0), Dose.bolus(amount=50.0, t=24.0)]

iv = Simulation(
    model=model,
    regime=regime,
    protocol="IV",
    clear=Clear(clearance=1.0),
)
solution = iv.solve(t_end=48.0)
iv.plot()
print(f"Computed {len(solution.t)} time points.")
plt.show()
