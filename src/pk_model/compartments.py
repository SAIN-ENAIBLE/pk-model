from dataclasses import dataclass
from math import isfinite

import numpy as np


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

    @property
    def volumes(self) -> np.ndarray:
        """[V_c, V_1, ..., V_N]: same order as the amounts in Model's y."""
        # TODO: central volume first, then each peripheral's volume.
        raise NotImplementedError

    @property
    def Q(self) -> np.ndarray:
        """[Q_1, ..., Q_N]: one transition rate per peripheral (no central entry).

        With no peripherals this should be an empty array, not an error.
        """
        # TODO
        raise NotImplementedError
