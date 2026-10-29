from collections.abc import Callable, Sequence

from numpy.typing import ArrayLike

from .clear import Clear
from .dose import Dose
from .model import Model
from .plotting import plot_solution
from .solve import SimulationSolution, solve_regimen


class Simulation:
    """Solve and plot IV or subcutaneous boluses (SC requires ``Ka``)."""

    def __init__(
        self,
        model: Model,
        regime: Sequence[Dose],
        protocol: str,
        clear: Clear,
        Ka: float | None = None,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a Model instance.")
        if isinstance(regime, (str, bytes)) or not isinstance(regime, Sequence):
            raise TypeError("regime must be a sequence of Dose events.")
        if not all(isinstance(dose, Dose) for dose in regime):
            raise TypeError("Every regime item must be a Dose instance.")
        route = model.validate_route(protocol, Ka)
        if not isinstance(clear, Clear):
            raise TypeError("clear must be a Clear instance.")

        self.model = model
        self.regime = tuple(regime)
        self.protocol = route
        self.Ka = Ka
        self.clear = clear
        self.solution: SimulationSolution | None = None

    def solve(
        self,
        t_end: float,
        t_start: float = 0.0,
        t_eval: ArrayLike | None = None,
        dose_rate: float | Callable[[float], float] = 0.0,
    ) -> SimulationSolution:
        """Apply boluses and an optional Dose(t) input rate over the time span.

        ``dose_rate`` enters central for IV and injection-site amount q_0 for SC.
        Use consistent amount/time units; the default rate is zero.
        """
        self.solution = solve_regimen(
            model=self.model,
            regimen=self.regime,
            clear=self.clear,
            t_span=(t_start, t_end),
            t_eval=t_eval,
            protocol=self.protocol,
            Ka=self.Ka,
            dose_rate=dose_rate,
        )
        return self.solution

    def plot(self):
        """Return a Matplotlib figure and axes for the solved regimen."""
        if self.solution is None:
            raise RuntimeError("Call solve() before plot().")
        return plot_solution(self.solution)
