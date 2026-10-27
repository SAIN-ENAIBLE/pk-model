import numpy as np


class CompartmentModel:
    """Linear compartmental pharmacokinetic model."""

    def __init__(
        self,
        central_volume: float,
        clearance: float,
        peripheral_volumes: list[float] | None = None,
        cp_clearances: list[float] | None = None,
    ):
        self.central_volume = central_volume
        self.clearance = clearance
        self.peripheral_volumes = peripheral_volumes or []
        self.cp_transfer = cp_transfers or []

        self._validate_parameters()

    def rhs(self, t, q):
        """Calculate the time derivatives of all compartment quantities."""

        q_c = q[0]
        q_peripheral = q[1:]

        c_c = q_c / self.central_volume

        dqdt = np.zeros_like(q, dtype=float)

        # Elimination from central compartment
        dqdt[0] = -self.clearance * c_c

        # Distribution between central and peripheral compartments
        for i, (q_p, v_p, Q) in enumerate(
            zip(
                q_peripheral,
                self.peripheral_volumes,
                self.cp_transfers,
            )
        ):
            c_p = q_p / v_p

            transfer = Q * (c_c - c_p)

            dqdt[0] -= transfer
            dqdt[i + 1] += transfer

        return dqdt

    def _validate_parameters(self):
        """Validate the pharmacokinetic model parameters."""

        if self.central_volume <= 0:
            raise ValueError("Central volume must be positive.")

        if self.clearance < 0:
            raise ValueError("Clearance cannot be negative.")

        if len(self.peripheral_volumes) != len(self.cp_transfers):
            raise ValueError(
                "Each peripheral compartment must have a corresponding central-peripheral transfer."
            )
        
        if any(volume <= 0 for volume in self.peripheral_volumes):
            raise ValueError(
                "Peripheral volumes must be positive."
            )

        if any(clearance < 0 for clearance in self.cp_clearances):
            raise ValueError(
            "Central-peripheral clearances cannot be negative."
            )