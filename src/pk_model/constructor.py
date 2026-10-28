import numpy as np
from numpy.typing import NDArray


class CompartmentModel:
    """Linear compartmental pharmacokinetic model.

    Construct a PK model consisting of one central compartment and
    zero or more peripheral compartments.

    Drug is eliminated from the central compartment and exchanged
    between the central and peripheral compartments.

    Parameters
    ----------
    central_volume : float
        Volume of the central compartment in L.
    clearance : float
        Clearance from the central compartment in L/h.
    peripheral_volumes : list[float] or None, optional
        Volumes of peripheral compartments in L.
        If None, a one-compartment model is constructed.
    intercompartmental_clearances : list[float] or None, optional
        Intercompartmental clearances between the central compartment
        and each peripheral compartment in L/h.

    Raises
    ------
    ValueError
        If volumes are non-positive, clearances are negative, or the
        number of peripheral volumes does not match the number of
        intercompartmental clearances.
    """

    def __init__(
        self,
        central_volume: float,
        clearance: float,
        peripheral_volumes: list[float] | None = None,
        intercompartmental_clearances: list[float] | None = None,
    ):
        self.central_volume = central_volume
        self.clearance = clearance
        self.peripheral_volumes = peripheral_volumes or []
        self.intercompartmental_clearances = (
            intercompartmental_clearances or []
        )

        self._validate_parameters()

    @property
    def n_peripheral(self) -> int:
        """Return the number of peripheral compartments."""
        return len(self.peripheral_volumes)

    @property
    def n_compartments(self) -> int:
        """Return the total number of compartments."""
        return 1 + self.n_peripheral

    def rhs(
        self,
        t: float,
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the ODE right-hand side.

        Parameters
        ----------
        t : float
            Current time in h. Required by ODE solvers such as
            ``scipy.integrate.solve_ivp``.
        q : numpy.ndarray
            Drug quantities in mg. The first element represents the
            central compartment; subsequent elements represent the
            peripheral compartments.

        Returns
        -------
        numpy.ndarray
            Time derivatives of compartment quantities in mg/h.

        Raises
        ------
        ValueError
            If the state vector does not contain one quantity for
            each compartment.
        """
        q = np.asarray(q, dtype=float)

        if len(q) != self.n_compartments:
            raise ValueError(
                f"Expected {self.n_compartments} state variables, "
                f"but received {len(q)}."
            )

        # Central compartment concentration
        q_c = q[0]
        c_c = q_c / self.central_volume

        # Initialise all differentials to zero
        dqdt = np.zeros(self.n_compartments, dtype=float)

        # Elimination from the central compartment
        dqdt[0] = -self.clearance * c_c

        # Exchange with each peripheral compartment
        for i, (v_p, q_inter) in enumerate(
            zip(
                self.peripheral_volumes,
                self.intercompartmental_clearances,
            )
        ):
            # q[0] is central, so peripheral compartment i is q[i + 1]
            q_p = q[i + 1]
            c_p = q_p / v_p

            # Net transfer from central -> peripheral
            transfer = q_inter * (c_c - c_p)

            # Whatever leaves central enters peripheral
            dqdt[0] -= transfer
            dqdt[i + 1] += transfer

        return dqdt

    def concentrations(
        self,
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert compartment quantities to concentrations.

        Parameters
        ----------
        q : numpy.ndarray
            Drug quantities in mg.

        Returns
        -------
        numpy.ndarray
            Drug concentrations in mg/L.
        """
        q = np.asarray(q, dtype=float)

        if len(q) != self.n_compartments:
            raise ValueError(
                f"Expected {self.n_compartments} state variables, "
                f"but received {len(q)}."
            )

        volumes = np.array(
            [self.central_volume, *self.peripheral_volumes],
            dtype=float,
        )

        return q / volumes

    def _validate_parameters(self) -> None:
        """Validate PK model parameters."""

        if self.central_volume <= 0:
            raise ValueError(
                "Central volume must be positive."
            )

        if self.clearance < 0:
            raise ValueError(
                "Clearance cannot be negative."
            )

        if len(self.peripheral_volumes) != len(
            self.intercompartmental_clearances
        ):
            raise ValueError(
                "Each peripheral compartment must have a corresponding "
                "intercompartmental clearance."
            )

        if any(
            volume <= 0
            for volume in self.peripheral_volumes
        ):
            raise ValueError(
                "Peripheral volumes must be positive."
            )

        if any(
            clearance < 0
            for clearance in self.intercompartmental_clearances
        ):
            raise ValueError(
                "Intercompartmental clearances cannot be negative."
            )