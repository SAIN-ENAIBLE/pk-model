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
    """Administration at the beginning of a simulation should be valid."""

    dose = Dose.bolus(
        amount=100.0,
        t=0.0,
    )

    assert dose.t == 0.0


# ---------------------------------------------------------------------
# Dose validation
# Reject non-positive/non-finite amounts and invalid administration times.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount",
    [0.0, -1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_dose_amount(amount):
    """Dose amount must be positive and finite."""

    with pytest.raises(ValueError):
        Dose.bolus(
            amount=amount,
            t=0.0,
        )


@pytest.mark.parametrize(
    "time",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_dose_time(time):
    """Dose time must be non-negative and finite."""

    with pytest.raises(ValueError):
        Dose.bolus(
            amount=100.0,
            t=time,
        )


# ---------------------------------------------------------------------
# Infusion
# Infusion dosing is explicitly not implemented yet.
# ---------------------------------------------------------------------


def test_infusion_not_implemented():
    """Infusion constructor should report that it is unsupported."""

    with pytest.raises(NotImplementedError):
        Dose.infusion(
            amount=100.0,
            t=0.0,
        )
