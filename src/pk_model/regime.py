"""
Administration regimes: how a dose gets into the central compartment.

Every regime answers two questions for Model.rhs():
- n_states: how many extra drug "buckets" does it add in front of the central
  compartment? (IV: 0. SC: 1, the depot under the skin.)
- derivatives(t, q_abs, dose_rate): given its own amounts and the current dose
  rate, return (how its own amounts change, drug flowing into central).
"""

import numpy as np


class Regime:
    """The contract every regime follows. Subclasses fill in both members."""

    n_states: int

    def derivatives(
        self, t: float, q_abs: np.ndarray, dose_rate: float
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class IV(Regime):
    """Dose goes straight into the central compartment."""

    n_states = 0

    def derivatives(
        self, t: float, q_abs: np.ndarray, dose_rate: float
    ) -> tuple[np.ndarray, float]:
        # TODO: IV owns no states, so what is "how my own amounts change"?
        #       And how much of dose_rate flows into central?
        raise NotImplementedError


class SC(Regime):
    """Dose goes into a depot, which feeds central at rate ka * depot. (Later.)

    da/dt = dose_rate - ka * a      (a = depot amount, the one state)
    inflow to central = ka * a
    """

    def __init__(self, ka: float) -> None:
        self.ka = ka
        self.n_states = 1

    def derivatives(
        self, t: float, q_abs: np.ndarray, dose_rate: float
    ) -> tuple[np.ndarray, float]:
        # TODO(later): after IV works.
        raise NotImplementedError
