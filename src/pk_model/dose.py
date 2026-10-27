"""Dose functions for pharmacokinetic modeling."""


def linear_dose(dose: float, time: float) -> float:
    """Calculate the linear dose at a given time.

    Args:
        dose (float): The administered dose.
        time (float): The time since administration.

    Returns:
        float: The calculated dose at the given time.
    """
    return dose * time


def bolus_dose(dose: float, time: float) -> float:
    """Calculate the bolus dose at a given time.

    Args:
        dose (float): The administered dose.
        time (float): The time since administration.

    Returns:
        float: The calculated dose at the given time.
    """
    return dose if time == 0 else 0.0