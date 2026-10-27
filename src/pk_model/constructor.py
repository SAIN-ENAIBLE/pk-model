class CompartmentModel(self):
    """Linear compartmental pharmacokinetic model."""

    def __init__(
        self,
        central_volume: float,
        clearance: float,
        peripheral_volumes: list[float]
        intercompartmental_clearances: list[float] |
    ):