from dataclasses import dataclass
from ._validate import require_positive
import numpy as np


@dataclass(frozen=True)
class Peripheral:
    """
    Container for peripheral volume and transition rates.
    
    volume: [ml]
    transition_rate: [ml][h-1]
    """
       
    volume: float
    transition_rate: float

    def __post_init__(self):
        require_positive("peripheral_volume", self.volume)
        require_positive("transition_rate", self.transition_rate)

@dataclass(frozen=True)
class Comp:
    """
    Container for all compartments (central and peripheral).
    
    Properties:
        volumes: Returns volume vector [V_c, V_1, ..., V_N].
        Q: Returns transition_rate vector [Q_1, ..., Q_N].
    """

    central_volume: float
    peripherals: tuple[Peripheral, ...] = ()

    def __post_init__(self):
        require_positive("central_volume",self.central_volume)

    @property
    def volumes(self) -> np.ndarray:
        """Property: comp.volumes = [V_c, V_1, ..., V_N]"""
        #TODO(Optimisation): rhs() calls thousands of times.
        #FIX 1: consider functools.cached_property
        #FIX 2: initialise once in post-init.
        
        peripheral_volumes = [periph.volume for periph in self.peripherals]
        all_volumes = np.array([self.central_volume] + peripheral_volumes)
        return all_volumes

    @property
    def Q(self) -> np.ndarray:
        """Property: comp.Q = [Q_1, ..., Q_N]."""
        
        peripheral_transitions = [periph.transition_rate for periph in self.peripherals]
        return np.array(peripheral_transitions)