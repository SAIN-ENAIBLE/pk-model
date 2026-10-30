# pk-model

A Python library for pharmacokinetic (PK) modelling: define a compartment
model, define a dosing protocol, solve, and plot.

[![CI](https://github.com/SAIN-ENAIBLE/pk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIN-ENAIBLE/pk-model/actions/workflows/ci.yml)

<!-- SPHINX-START -->

## Installation

```bash
pip install pk-model
```

## Administration routes

`Simulation` supports intravenous (`"IV"`) and subcutaneous (`"SC"`) boluses.
For SC, supply a positive absorption rate `Ka` in inverse-time units:

```python
from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.simulation import Simulation

simulation = Simulation(
    model=Model(
        CentralParams(
            central_volume=5.0,
            clearance=1.0,
            peripherals=(PeripheralParams(volume=10.0, flow=2.0),),
        )
    ),
    regime=[Dose.bolus(amount=100.0, t=0.0)],
    protocol="SC",
    clear=Clear(clearance=1.0),
    Ka=0.5,
)
solution = simulation.solve(t_end=24.0)
```

SC doses enter the injection-site compartment (amount `depot`) and absorb into
the central compartment with complete bioavailability. Solution rows contain
central, peripheral, then `depot` amounts, identified by `solution.depot_index`.
At injection times, `solution.y` contains post-dose amounts. Pre-dose amounts
are retained in `solution.pre_dose_states`, allowing plots to show vertical
bolus jumps while keeping `solution.t` strictly increasing.
For one peripheral, the equations are:

```text
d(depot)/dt = Dose(t) - Ka*depot
dqc/dt      = Ka*depot - CL*qc/Vc - Qp1*(qc/Vc - qp1/Vp1)
dqp1/dt     = Qp1*(qc/Vc - qp1/Vp1)
```

`Model.rhs(..., protocol="SC", Ka=0.5)` implements these equations, extending
the exchange terms to every configured peripheral. States are ordered
`[qc, qp1, ..., depot]`.
Use `protocol="IV"` (the default) for direct central input without `depot`.
`Ka` is in 1/h when time is in hours; `Dose(t)` is an amount per hour.
Bolus events add their amounts to `depot` at the administration times, with zero
input rate between events by default. An optional `dose_rate` supplies the
continuous `Dose(t)` term, as a constant or a function of time:

```python
solution = simulation.solve(t_end=24.0, dose_rate=lambda t: 2.0 * t)
```

This rate adds to any boluses in `regime`; use `regime=[]` for continuous input
alone. For IV, `dose_rate` enters the central compartment directly.

Full route names are also accepted, case-insensitively. Intramuscular (`"IM"`)
and intradermal (`"ID"`) routes raise `NotImplementedError` in both `Simulation`
and the older `Protocol` API.

## Examples

Run the example scripts from the repository root:

```bash
uv run python scripts/example.py   # One peripheral: compare IV and SC
uv run python scripts/2_periph.py  # Two peripherals: repeated IV doses
uv run python scripts/3_periph.py  # Compare two and three peripherals with the same IV regimen
```

Each model also has a central compartment; SC adds an absorption depot.
The scripts print final amounts and display comparison plots.

## Development

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                  # create .venv with dev dependencies
uv run pytest --cov      # run the tests with coverage
uv run ruff check        # lint
uv run ruff format       # format
uv run --group docs sphinx-build -n -W -b html docs docs/_build/html  # build the docs
```

Optionally install the pre-commit hooks (ruff on every commit):

```bash
uvx pre-commit install
```

## License
BSD-3.
