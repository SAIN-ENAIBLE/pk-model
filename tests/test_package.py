from __future__ import annotations

import importlib.metadata

import pk_model as m


def test_version() -> None:
    assert importlib.metadata.version("pk_model") == m.__version__
