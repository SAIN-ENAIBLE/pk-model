# Imports

import numpy as np
import pytest

from pk_model.routes import validate_route

### Tests

# ---------------------------------------------------------------------
# Route normalisation
# Check abbreviations and full route names map to canonical names.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("IV", "IV"),
        ("iv", "IV"),
        ("intravenous", "IV"),
        ("SC", "SC"),
        ("sc", "SC"),
        ("subcutaneous", "SC"),
    ],
)
def test_supported_route_normalisation(route, expected):
    """Supported route names should be normalised."""

    ka = 0.5 if expected == "SC" else None

    assert validate_route(route, ka) == expected


# ---------------------------------------------------------------------
# Unsupported routes
# IM and ID are recognised but deliberately not implemented.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "route",
    [
        "IM",
        "im",
        "intramuscular",
        "ID",
        "id",
        "intradermal",
    ],
)
def test_recognised_but_unimplemented_routes(route):
    """IM and ID should explicitly report that they are unsupported."""

    with pytest.raises(NotImplementedError):
        validate_route(route, None)


# ---------------------------------------------------------------------
# Unknown routes
# Reject route names outside the recognised administration routes.
# ---------------------------------------------------------------------


def test_unknown_route():
    """Unknown administration route should raise ValueError."""

    with pytest.raises(ValueError):
        validate_route("banana", None)


def test_route_must_be_string():
    """Route identifier should be a string."""

    with pytest.raises(TypeError):
        validate_route(
            protocol=123,
            Ka=None,
        )


# ---------------------------------------------------------------------
# SC absorption parameter validation
# SC requires a positive finite first-order absorption rate Ka.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ka",
    [None, 0.0, -1.0, np.inf, np.nan],
)
def test_sc_requires_valid_ka(ka):
    """SC should require positive finite Ka."""

    with pytest.raises(ValueError):
        validate_route("SC", ka)
