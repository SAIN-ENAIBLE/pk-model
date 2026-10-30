from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from .clear import Clear
from .dose import Dose
from .model import Model


@dataclass(frozen=True)
class SimulationSolution:
    """Amounts ordered central, peripherals, then depot for SC solutions.

    ``y`` contains post-dose amounts at bolus times. ``pre_dose_states``
    retains the amounts immediately before each bolus for plotting jumps.
    """

    t: np.ndarray
    y: np.ndarray
    success: bool = True
    message: str = (
        "The solver successfully reached the end of the integration interval."
    )
    depot_index: int | None = None
    pre_dose_states: dict[float, np.ndarray] = field(default_factory=dict)


def solve_regimen(
    model: Model,
    regimen: Sequence[Dose],
    clear: Clear,
    t_span: tuple[float, float],
    t_eval: ArrayLike | None = None,
    protocol: str = "IV",
    Ka: float | None = None,
    dose_rate: float | Callable[[float], float] = 0.0,
    *,
    initial_state: ArrayLike | None = None,
) -> SimulationSolution:
    """Apply IV boluses to central or SC boluses to the injection-site depot.

    ``Ka`` has inverse-time units. SC assumes complete bioavailability.
    ``dose_rate`` supplies an additional constant or callable input rate:
    into central for IV, into depot for SC. Boluses remain discrete jumps.
    ``initial_state`` sets amounts before any dose at the simulation start,
    ordered central, peripherals, then depot for SC. The default is all zeros.
    """
    route = model.validate_route(protocol, Ka)
    has_depot = route == "SC"
    dose_index = model.n_compartments() if has_depot else 0

    def rhs(t, y):
        return model.rhs(t, y, dose_rate=dose_rate, clear=clear, protocol=route, Ka=Ka)

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

    state_count = model.n_compartments() + has_depot
    state = (
        np.zeros(state_count, dtype=float)
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    if (
        state.shape != (state_count,)
        or not np.all(np.isfinite(state))
        or np.any(state < 0)
    ):
        raise ValueError(
            f"initial_state must contain {state_count} non-negative finite amounts."
        )
    pre_dose_states = {}
    if start in dose_totals:
        pre_dose_states[start] = state.copy()
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
        if len(solution.t) == 0 or solution.t[-1] != boundary:
            raise RuntimeError(
                f"ODE solver did not return the integration endpoint {boundary}."
            )
        if not np.all(np.isfinite(solution.y)):
            raise RuntimeError("ODE solver returned non-finite compartment amounts.")

        result_times.extend(solution.t[:-1].tolist())
        result_states.extend(solution.y[:, :-1].T.tolist())
        # Keep the integrated pre-dose result intact; restart from a separate
        # state containing the instantaneous jump.
        state = solution.y[:, -1].copy()
        if boundary in dose_totals:
            pre_dose_states[boundary] = state.copy()
        state[dose_index] += dose_totals.get(boundary, 0.0)
        result_times.append(boundary)
        result_states.append(state.copy())
        current_time = boundary

    return SimulationSolution(
        t=np.asarray(result_times),
        y=np.asarray(result_states, dtype=float).T,
        depot_index=dose_index if has_depot else None,
        pre_dose_states=pre_dose_states,
    )


def _validate_time_span(t_start: float, t_end: float) -> None:
    if not isfinite(t_start) or not isfinite(t_end) or t_start >= t_end:
        raise ValueError("Time span must have finite start < end.")
