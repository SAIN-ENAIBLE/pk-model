# Imports

import numpy as np
import pytest

from pk_model.dose import Dose

### Tests

# ---------------------------------------------------------------------
# Bolus construction
# Check that valid bolus dose events are represented correctly.
# ---------------------------------------------------------------------


def test_bolus_dose():
    """Bolus should store amount and administration time."""

    dose = Dose.bolus(
        amount=100.0,
        t=10.0,
    )

    assert dose.amount == 100.0
    assert dose.t == 10.0


def test_bolus_at_time_zero():
    """A bolus at the start of a simulation should be valid."""

    dose = Dose.bolus(
        amount=100.0,
        t=0.0,
    )

    assert dose.amount == 100.0
    assert dose.t == 0.0


# ---------------------------------------------------------------------
# Dose amount validation
# Reject zero, negative, and non-finite dose amounts.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount",
    [0.0, -1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_dose_amount_raises_value_error(amount):
    """Dose amount must be positive and finite."""

    with pytest.raises(ValueError):
        Dose.bolus(
            amount=amount,
            t=0.0,
        )


# ---------------------------------------------------------------------
# Dose time validation
# Reject negative and non-finite administration times.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "time",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_dose_time_raises_value_error(time):
    """Dose time must be non-negative and finite."""

    with pytest.raises(ValueError):
        Dose.bolus(
            amount=100.0,
            t=time,
        )


# ---------------------------------------------------------------------
# Infusion dosing
# Confirm that the currently unsupported infusion API fails explicitly.
# ---------------------------------------------------------------------


def test_infusion_not_implemented():
    """Infusion dosing should explicitly report that it is unsupported."""

    with pytest.raises(NotImplementedError):
        Dose.infusion(
            amount=100.0,
            t=0.0,
        )
