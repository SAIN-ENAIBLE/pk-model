# pk-model

A Python library for pharmacokinetic (PK) modelling: define a compartment
model, define a dosing protocol, solve, and plot.

[![CI](https://github.com/org/pk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/org/pk-model/actions/workflows/ci.yml)

<!-- SPHINX-START -->

## Installation

```bash
pip install .
```

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
