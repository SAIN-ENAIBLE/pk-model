# Imports

import numpy as np
import pytest

from pk_model.clear import Clear

# Tests

# ---------------------------------------------------------------------
# Clearance construction
# Check that only non-negative finite clearance values are accepted.
# ---------------------------------------------------------------------


def test_valid_clearance():
    """Valid clearance should be stored correctly."""

    clear = Clear(clearance=2.0)

    assert clear.clearance == 2.0


def test_zero_clearance_disables_elimination():
    assert Clear(clearance=0.0).linear(100.0) == 0.0


@pytest.mark.parametrize(
    "clearance",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_clearance_raises_value_error(clearance):
    """Clearance must be non-negative and finite."""

    with pytest.raises(ValueError):
        Clear(clearance=clearance)


# ---------------------------------------------------------------------
# Linear clearance
# Check elimination rate equals clearance multiplied by concentration.
# ---------------------------------------------------------------------


def test_linear_clearance():
    """Linear elimination should equal CL multiplied by concentration."""

    clear = Clear(clearance=2.0)

    actual = clear.linear(concentration=5.0)

    expected = 10.0

    assert actual == expected


def test_zero_concentration_gives_zero_elimination():
    """No drug concentration should produce no elimination."""

    clear = Clear(clearance=2.0)

    assert clear.linear(0.0) == 0.0
