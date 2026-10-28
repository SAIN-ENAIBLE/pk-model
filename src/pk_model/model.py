from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .compartments import Comp
from .regime import Regime


@dataclass(frozen=True)
class Model:
    """The only place that knows the whole equation. Everything else answers
    one narrow question for it.

        dq_c/dt = inflow - clear(t, c_c) - sum_i Q_i (c_c - c_i)
        dq_i/dt = Q_i (c_c - c_i)            (one per peripheral i)

    with concentration c = q / V. The state vector is
    y = [regime states..., q_c, q_p1, q_p2, ...].
    """

    dose: Callable[[float], float]  # dose(t) -> amount / time
    regime: Regime  # supplies "inflow" and any extra states
    comp: Comp  # supplies volumes and Q as arrays
    clear: Callable[[float, float], float]  # clear(t, conc) -> amount / time

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Return dy/dt at time t. solve_ivp calls this over and over."""
        # 1. Split y into the regime's own states and [q_c, q_p1, ...].
        #    Hint: self.regime.n_states says where the split goes.
        # 2. Concentrations of central + peripherals: c = q / V.
        #    Hint: self.comp.volumes is an array, so divide the whole array at once.
        # 3. exchange_i = Q_i * (c_c - c_i), all peripherals at once.
        #    Hint: c_c is conc[0], the c_i are conc[1:]. self.comp.Q is an array.
        # 4. Ask the regime: (dq_abs, inflow) = regime.derivatives(t, q_abs, self.dose(t))
        # 5. dq_c = inflow - clear(t, c_c) - (sum of exchange)
        # 6. Glue the pieces back together in the SAME order as y:
        #    [dq_abs..., dq_c, exchange...]   (np.concatenate)
        raise NotImplementedError

    def solve(self, t_end: float, y0: np.ndarray | None = None):
        """Integrate from t=0 to t_end. Returns solve_ivp's result (.t, .y)."""
        # 1. How long is y? regime states + 1 central + number of peripherals.
        # 2. If y0 is None, start with no drug anywhere (zeros of that length).
        # 3. return solve_ivp(self.rhs, (0, t_end), y0, rtol=..., atol=...)
        #    The test compares to 1e-5, so the solver needs tighter tolerances
        #    than its defaults (try rtol=1e-8, atol=1e-10).
        raise NotImplementedError
