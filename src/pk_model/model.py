from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

from .clear import Clear


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
    order given by ``params.peripherals``, then depot for SC administration.
    ``params.clearance`` is the central
    clearance; each peripheral clearance is its intercompartmental clearance.
    Each peripheral exchanges drug only with the central compartment.
    Amounts, volumes, clearances, and time must use consistent units.
    """

    def __init__(self, params: CentralParams) -> None:
        if not isinstance(params, CentralParams):
            raise TypeError("params must be a CentralParams instance.")
        self.params = params

    def n_compartments(self) -> int:
        """Return the number of central and peripheral compartments, excluding depot."""
        return 1 + len(self.params.peripherals)

    @staticmethod
    def validate_route(protocol: str, Ka: float | None = None) -> str:
        """Normalise a route and validate its absorption parameters."""
        if not isinstance(protocol, str):
            raise TypeError("protocol must be a route string.")
        route = protocol.upper()
        route = {
            "INTRAVENOUS": "IV",
            "SUBCUTANEOUS": "SC",
            "INTRAMUSCULAR": "IM",
            "INTRADERMAL": "ID",
        }.get(route, route)
        if route in {"IM", "ID"}:
            raise NotImplementedError(f"The {route} protocol is not implemented.")
        if route not in {"IV", "SC"}:
            raise ValueError(f"Unknown administration route: {protocol!r}.")
        if route == "SC" and (Ka is None or not isfinite(Ka) or Ka <= 0):
            raise ValueError("SC requires a positive finite absorption rate Ka.")
        return route

    def rhs(
        self,
        t: float,
        y: Sequence[float],
        dose_rate: float | Callable[[float], float] = 0.0,
        clear: Clear | None = None,
        *,
        protocol: str = "IV",
        Ka: float | None = None,
    ) -> list[float]:
        """Return amount derivatives for IV or SC administration.

        IV states: ``[q_c, q_p1, ..., q_pN]``.
        SC states: ``[q_c, q_p1, ..., q_pN, depot]``.

        ``dose_rate`` is Dose(t), a constant or callable input in amount/time.
        It enters central for IV and depot for SC. SC absorption transfers
        Ka*depot to central, with d(depot)/dt = Dose(t) - Ka*depot. ``Ka`` is a
        positive absorption rate in inverse-time units.

        Each peripheral receives Q_i*(q_c/V_c - q_pi/V_pi). Central loses
        these exchanges and CL*q_c/V_c through elimination.

        When ``clear`` is provided, its linear clearance replaces the clearance
        value in ``params`` for this rate calculation.

        Negative states are allowed because numerical solvers may briefly
        step below zero; states are never clipped within the equations.
        """
        route = self.validate_route(protocol, Ka)
        is_sc = route == "SC"
        compartment_count = self.n_compartments()
        expected_states = compartment_count + int(is_sc)

        if len(y) != expected_states:
            raise ValueError(
                f"{route} requires {expected_states} states; received {len(y)}."
            )

        input_rate = dose_rate(t) if callable(dose_rate) else dose_rate
        if not isfinite(input_rate) or input_rate < 0:
            raise ValueError("Dose rate must be a non-negative finite number.")

        p = self.params
        central_concentration = y[0] / p.central_volume
        elimination_rate = (
            clear.linear(central_concentration)
            if clear is not None
            else p.clearance * central_concentration
        )
        d_depot = 0.0
        if is_sc:
            depot = y[-1]
            if Ka is None or not isfinite(Ka) or Ka <= 0:
                raise ValueError("SC requires a positive finite absorption rate Ka.")
            absorption = Ka * depot
            central_input = absorption
            d_depot = input_rate - absorption
        else:
            central_input = input_rate

        central_rate = central_input - elimination_rate
        peripheral_rates = []

        peripheral_amounts = y[1:compartment_count]
        for peripheral, amount in zip(p.peripherals, peripheral_amounts, strict=True):
            concentration = amount / peripheral.volume
            exchange = peripheral.flow * (central_concentration - concentration)
            central_rate -= exchange
            peripheral_rates.append(exchange)

        rates = [central_rate, *peripheral_rates]
        if is_sc:
            rates.append(d_depot)
        return rates

    def rhs_sc(
        self,
        t: float,
        y: Sequence[float],
        Ka: float,
        dose_rate: float | Callable[[float], float] = 0.0,
        clear: Clear | None = None,
    ) -> list[float]:
        """Return SC derivatives for central, peripherals, and depot amounts."""
        return self.rhs(t, y, dose_rate=dose_rate, clear=clear, protocol="SC", Ka=Ka)
