from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Dose:
    """A dose event with a kind, amount, and administration time."""

    amount: float
    t: float

    def __post_init__(self):
        if not isfinite(self.amount) or self.amount <= 0:
            raise ValueError("Dose amount must be positive finite.")
        if not isfinite(self.t) or self.t < 0:
            raise ValueError("Dose time must be non-negative finite.")

    @classmethod
    def bolus(cls, amount: float, t: float) -> "Dose":
        """Create a bolus dose event."""
        return cls(amount=amount, t=t)

    @classmethod
    def infusion(cls, amount: float, t: float) -> "Dose":
        """Create an infusion dose event."""
        raise NotImplementedError("Infusion dose not implemented!")
