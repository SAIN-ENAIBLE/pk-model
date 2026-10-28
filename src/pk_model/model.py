from dataclasses import dataclass

from .dose import *
from .compartments import Comp
@dataclass
class Model:
    dose: Dose
    regime: Regime
    comp: Comp
    clear: Clear