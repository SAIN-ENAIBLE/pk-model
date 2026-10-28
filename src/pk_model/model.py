from collections.abc import Callable, Sequence
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


class Model:
    """Linear compartment model with elimination from the central compartment.

    States are drug amounts, ordered central first, then peripherals in the
    order given by ``params.peripherals``. ``params.clearance`` is the central
    clearance; each peripheral clearance is its intercompartmental clearance.
    Each peripheral exchanges drug only with the central compartment.
    Amounts, volumes, clearances, and time must use consistent units.
    """

    def __init__(self, params: CentralParams) -> None:
        if not isinstance(params, CentralParams):
            raise TypeError("params must be a CentralParams instance.")
        self.params = params

    def n_compartments(self) -> int:
        """Return the number of compartments in the model."""
        return 1 + len(self.params.peripherals)

    def rhs(self, t: float, y: Sequence[float], dose_rate: float | Callable[[float], float] = 0.0) -> list[float]:
        """Return drug amount derivatives at time ``t``.

        ``y`` contains ``[q_c, q_p1, ..., q_pN]``. ``dose_rate`` is ``Dose(t)``:
        drug entering the central compartment in amount per unit time, either
        constant or a callable evaluated at ``t``. Its default is zero, so
        ``rhs(t, y)`` can be passed directly to an ODE solver. A bolus adds its
        amount directly to the central state at the administration time.

        For each peripheral i, exchange_i = Q_i * (q_c / V_c - q_i / V_i).
        Then dq_c/dt = Dose(t) - CL * q_c / V_c - sum(exchange_i), and
        dq_i/dt = exchange_i.

        Negative states are allowed because numerical solvers may briefly
        step below zero; states are never clipped within the equations.
        """
        if len(y) != self.n_compartments():
            raise ValueError("State vector length does not match number of compartments.")

        input_rate = dose_rate(t) if callable(dose_rate) else dose_rate
        if not isfinite(input_rate) or input_rate < 0:
            raise ValueError("Dose rate must be a non-negative finite number.")

        p = self.params
        central_concentration = y[0] / p.central_volume
        central_rate = input_rate - p.clearance * central_concentration
        peripheral_rates = []

        for peripheral, amount in zip(p.peripherals, y[1:], strict=True):
            concentration = amount / peripheral.volume
            exchange = peripheral.flow * (central_concentration - concentration)
            central_rate -= exchange
            peripheral_rates.append(exchange)

        return [central_rate, *peripheral_rates]
