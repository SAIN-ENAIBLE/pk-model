from collections.abc import Callable
from ._validate import require_non_negative, require_positive

def infusion(start, end, rate) -> Callable[[float],float]:
    """
    Returns continuous infusion dose function which outputs rate within the interval, otherwise 0.

    Args:
        start (_type_): Dose start time (inclusive).
        end (_type_): Dose end time (exclusive).
        rate (_type_): Dose rate in [ng][h-1].

    Returns:
        Callable[[float],float]: Infusion dose function.
    """
    
    require_positive("rate", rate)
    
    def dose(t) -> float:
        require_non_negative("time", t)
        if start <= t < end:
            return rate
        return 0
    
    return dose

def bolus(amount: float, time: float) -> Callable[[float],float]:
    """
    Returns instantaneous bolus function which outputs a single dose spike at the specified time.

    Args:
        amount (float): Dose quantity in [ng].
        time (float): Time when instantaneous dose is delivered.

    Returns:
        Callable[[float],float]: Instantaenous bolus function.
    """
    
    def dose(t) -> float:
        require_non_negative("time",t)
        if t == time:
            return amount
        return 0
        
    return dose
