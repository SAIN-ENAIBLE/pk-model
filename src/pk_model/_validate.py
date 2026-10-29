from math import isfinite


def require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <=0:
        raise ValueError(f"{name} must be a positive finite number (got {value} instead).")
    
def require_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative (got {value} instead).")
