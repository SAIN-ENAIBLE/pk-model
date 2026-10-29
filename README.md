# pk-model

A Python library for pharmacokinetic (PK) modelling: define a compartment
model, define a dosing protocol, solve, and plot.

[![CI](https://github.com/org/pk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/org/pk-model/actions/workflows/ci.yml)

<!-- SPHINX-START -->

## Installation

```bash
pip install .
```

## Administration routes

`Simulation` supports intravenous (`"IV"`) and subcutaneous (`"SC"`) boluses.
For SC, supply a positive absorption rate `Ka` in inverse-time units:

```python
from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model
from pk_model.simulation import Simulation

simulation = Simulation(
    model=Model(CentralParams(central_volume=5.0, clearance=1.0)),
    regime=[Dose.bolus(amount=100.0, t=0.0)],
    protocol="SC",
    clear=Clear(clearance=1.0),
    Ka=0.5,
)
solution = simulation.solve(t_end=24.0)
```

SC doses enter a depot and absorb into the central compartment with complete
bioavailability. Solution rows contain central, peripheral, then depot amounts.
Full route names are also accepted, case-insensitively. Intramuscular (`"IM"`)
and intradermal (`"ID"`) routes raise `NotImplementedError` in both `Simulation`
and the older `Protocol` API.

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
