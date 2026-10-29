from collections.abc import Callable
from math import isfinite

def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <=0:
        raise ValueError(f"{name} must be a positive finite number (got {value} instead).")

def constant(rate: float) -> Callable[[float, float], float]:
    """
    Returns a concentration-independent function with uniform clearance.

    Args:
        rate (float): Constant clearance rate in [ng][h-1].

    Returns:
        Callable[[float, float], float]: Constant clearance rate function.
    """
    
    
    def clear(_t:float, _conc: float) -> float:
        #_t and _conc are dummy variables.
        return rate
    
    return clear

def linear(CL: float) -> Callable[[float, float], float]:
    """
    Returns a function of concentration with linear clearance rate.
    
    Units: CL[ml][h-1] * conc[ng][ml] = rate[ng][h-1]

    Args:
        CL (float): Linear clearance rate in [ml][h-1].

    Returns:
        Callable[[float, float], float]: Linear clearance rate function.
    """
    
    _require_positive("CL", CL)

    def clear(_t: float, conc: float) -> float:
        rate = CL * conc
        return rate
    
    return clear

def michaelis_menten(Vmax: float, Km: float) -> Callable[[float, float], float]:
    """
    Returns a function of concentration with Michealis-Menten saturation kinetics.

    Args:
        Vmax (float): Maximal velocity (full saturation).
        Km (float): Substrate concentration corresponding to a rate of 0.5*Vmax.

    Returns:
        Callable[[float, float], float]: Michealis-Menten clearance rate function.
    """

    _require_positive("Vmax", Vmax)
    _require_positive("Km", Km)
    
    def clear(_t: float, conc: float) -> float:
        
        rate = (Vmax * conc)/(Km + conc)
        return rate
    
    return clear