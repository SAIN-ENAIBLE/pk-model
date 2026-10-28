from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PeripheralParams:
    """Peripheral parameters for the model."""

    volume: float
    flow: float

    def __post_init__(self):
        if not isfinite(self.volume) or self.volume <= 0:
            raise ValueError("Volume must be a positive finite number.")
        if not isfinite(self.flow) or self.flow <= 0:
            raise ValueError("Flow must be a positive finite number.")


@dataclass(frozen=True)
class CentralParams:
    """Central parameters for the model."""

    central_volume: float
    clearance: float
    peripherals: tuple[PeripheralParams, ...] = ()

    def __post_init__(self):
        if not isfinite(self.central_volume) or self.central_volume <= 0:
            raise ValueError("Central volume must be a positive finite number.")
        if not isfinite(self.clearance) or self.clearance <= 0:
            raise ValueError("Clearance must be a positive finite number.")