from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.integrate import solve_ivp

from .clear import Clear
from .dose import Dose
from .model import Model
from .routes import validate_route


@dataclass(frozen=True)
class SimulationSolution:
    """Amounts ordered central, peripherals, then depot for SC solutions."""

    t: np.ndarray
    y: np.ndarray
    success: bool = True
    message: str = (
        "The solver successfully reached the end of the integration interval."
    )
    depot_index: int | None = None


def solve_regimen(
    model: Model,
    regimen: Sequence[Dose],
    clear: Clear,
    t_span: tuple[float, float],
    t_eval: Sequence[float] | None = None,
    protocol: str = "IV",
    Ka: float | None = None,
) -> SimulationSolution:
    """Apply IV boluses to central or SC boluses to a first-order depot.

    ``Ka`` has inverse-time units. SC assumes complete bioavailability.
    """
    route = validate_route(protocol, Ka)
    has_depot = route == "SC"
    dose_index = model.n_compartments() if has_depot else 0

    def rhs(t, y):
        if has_depot:
            absorption = Ka * y[-1]
            rates = model.rhs(t, y[:-1], dose_rate=absorption, clear=clear)
            return [*rates, -absorption]
        return model.rhs(t, y, clear=clear)

    start, end = t_span
    _validate_time_span(start, end)
    dose_totals: dict[float, float] = {}
    for dose in regimen:
        if not start <= dose.t <= end:
            raise ValueError("All dose times must be within the simulation time span.")
        dose_totals[dose.t] = dose_totals.get(dose.t, 0.0) + dose.amount

    if t_eval is None:
        output_times = np.linspace(start, end, 200)
    else:
        output_times = np.asarray(t_eval, dtype=float)
        if output_times.ndim != 1 or len(output_times) == 0:
            raise ValueError("t_eval must be a non-empty one-dimensional sequence.")
        if (
            not np.all(np.isfinite(output_times))
            or output_times[0] < start
            or output_times[-1] > end
            or np.any(np.diff(output_times) <= 0)
        ):
            raise ValueError("t_eval must be finite, increasing, and within t_span.")

    output_times = np.unique(
        np.concatenate((output_times, np.fromiter(dose_totals, dtype=float)))
    )
    boundaries = sorted({end, *dose_totals.keys()})

    state = np.zeros(model.n_compartments() + has_depot, dtype=float)
    state[dose_index] += dose_totals.get(start, 0.0)
    result_times = [start]
    result_states = [state.copy()]
    current_time = start

    for boundary in boundaries:
        if boundary <= start:
            continue

        segment_times = output_times[
            (output_times > current_time) & (output_times < boundary)
        ]
        evaluation_times = np.append(segment_times, boundary)
        solution = solve_ivp(
            fun=rhs,
            t_span=(current_time, boundary),
            y0=state,
            t_eval=evaluation_times,
            rtol=1e-9,
            atol=1e-11,
        )

        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")

        result_times.extend(solution.t[:-1].tolist())
        result_states.extend(solution.y[:, :-1].T.tolist())
        state = solution.y[:, -1]
        state[dose_index] += dose_totals.get(boundary, 0.0)
        result_times.append(boundary)
        result_states.append(state.copy())
        current_time = boundary

    return SimulationSolution(
        t=np.asarray(result_times),
        y=np.asarray(result_states, dtype=float).T,
        depot_index=dose_index if has_depot else None,
    )


def _validate_time_span(t_start: float, t_end: float) -> None:
    if not isfinite(t_start) or not isfinite(t_end) or t_start >= t_end:
        raise ValueError("Time span must have finite start < end.")


def solve_protocol(protocol, t_span, t_eval):
    """Solve a Protocol's compartment equations over the requested time span."""
    y0 = [protocol.central["qc"]] + protocol.qp_list

    if protocol.type != "IV":
        y0.append(0.0 if callable(protocol.dose) else protocol.dose)

    return solve_ivp(
        fun=protocol.system_eq,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )
