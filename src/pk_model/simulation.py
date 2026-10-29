from collections.abc import Sequence

from .clear import Clear
from .dose import Dose
from .model import Model
from .plotting import plot_solution
from .routes import validate_route
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
        route = validate_route(protocol, Ka)
        if not isinstance(clear, Clear):
            raise TypeError("clear must be a Clear instance.")

        self.model = model
        self.regime = tuple(regime)
        self.protocol = route
        self.Ka = Ka
        self.clear = clear
        self.solution: SimulationSolution | None = None

    def solve(
        self, t_end: float, t_start: float = 0.0, t_eval: Sequence[float] | None = None
    ) -> SimulationSolution:
        """Solve from ``t_start`` to ``t_end``, applying boluses at their times."""
        self.solution = solve_regimen(
            model=self.model,
            regimen=self.regime,
            clear=self.clear,
            t_span=(t_start, t_end),
            t_eval=t_eval,
            protocol=self.protocol,
            Ka=self.Ka,
        )
        return self.solution

    def plot(self):
        """Return a Matplotlib figure and axes for the solved regimen."""
        if self.solution is None:
            raise RuntimeError("Call solve() before plot().")
        return plot_solution(self.solution)
