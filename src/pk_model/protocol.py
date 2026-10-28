from dataclasses import dataclass

import numpy as np

"""mg for the dose, ng for the amount, hours for time, L for volume, L/h for clearance and flow."""


@dataclass(frozen=True)
class Bolus:
    time: float
    amount: float

    def __post_init__(self):
        if not np.isfinite(self.time):
            raise ValueError("Bolus time must be a finite number.")
        if not np.isfinite(self.amount) or self.amount <= 0:
            raise ValueError("Bolus amount must be a positive finite number.")


@dataclass(frozen=True)
class Infusion:
    start: float
    end: float
    rate: float  # Amount per unit time

    def __post_init__(self):
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("Infusion start and end times must be finite numbers.")
        if self.start >= self.end:
            raise ValueError("Infusion start time must be less than end time.")
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("Infusion rate must be a positive finite number.")


@dataclass(frozen=True)
class Protocol:
    """Dosing protocol for a pharmacokinetic model.

    ``boluses`` is a list of bolus doses, each with a time and amount.
    ``infusions`` is a list of infusions, each with a start time, end time, and rate.
    """

    boluses: tuple[Bolus, ...] = ()
    infusions: tuple[Infusion, ...] = ()
