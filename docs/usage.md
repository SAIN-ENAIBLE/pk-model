# Usage

## IV bolus simulation

First construct a pharmacokinetic model:

```python
from pk_model.model import CentralParams, Model

model = Model(
    CentralParams(
        central_volume=5.0,
        clearance=1.0,
    )
)
```

Define a dose:

```python
from pk_model.dose import Dose

doses = [
    Dose.bolus(amount=100.0, t=0.0),
]
```

Define the clearance:

```python
from pk_model.clear import Clear

clearance = Clear(clearance=1.0)
```

Create the simulation:

```python
from pk_model.simulation import Simulation

simulation = Simulation(
    model=model,
    regime=doses,
    protocol="IV",
    clear=clearance,
)
```

Solve it:

```python
solution = simulation.solve(t_end=24.0)
```

Plot the result:

```python
simulation.plot()
```