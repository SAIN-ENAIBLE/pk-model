from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Peripheral:
    """Peripheral parameters for the model."""

    volume: float
    transition_rate: float

    def __post_init__(self):
        if not isfinite(self.volume) or self.volume <= 0:
            raise ValueError("Volume must be a positive finite number.")
        if not isfinite(self.transition_rate) or self.transition_rate <= 0:
            raise ValueError("Flow must be a positive finite number.")


@dataclass(frozen=True)
class Comp:
    """All model compartments (central + peripherals)."""

    central_volume: float
    peripherals: tuple[Peripheral, ...] = ()

    def __post_init__(self):
        if not isfinite(self.central_volume) or self.central_volume <= 0:
            raise ValueError("Central volume must be a positive finite number.")