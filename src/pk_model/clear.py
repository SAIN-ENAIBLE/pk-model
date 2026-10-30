from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Clear:
    """Linear central clearance; a value of zero disables elimination."""

    clearance: float

    def __post_init__(self):
        if not isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("Clearance must be a non-negative finite number.")

    def linear(self, concentration: float) -> float:
        """Calculate the elimination rate using linear clearance."""
        return self.clearance * concentration

    def michaelis_menten(self, vmax: float, km: float) -> float:
        """Calculate the elimination rate using Michaelis-Menten kinetics."""
        raise NotImplementedError("Michaelis-Menten clearance is not implemented!")
