from math import isfinite

from scipy.integrate import solve_ivp

from .model import Model
from .protocol import Protocol


def solve(model: Model, time_span, protocol: Protocol):
    """Solve an initially drug-free model with one IV bolus at the start."""
    if len(protocol.boluses) != 1 or protocol.infusions:
        raise ValueError("This solver supports exactly one bolus and no infusions.")

    start, end = time_span
    if not isfinite(start) or not isfinite(end) or start >= end:
        raise ValueError("Time span must have finite start < end.")

    bolus = protocol.boluses[0]

    if bolus.time != start:
        raise ValueError("Bolus time must match the start of the time span.")

    y0 = [0.0] * model.n_compartments()
    y0[0] = bolus.amount

    sol = solve_ivp(fun=model.rhs, t_span=time_span, y0=y0)

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    return sol
