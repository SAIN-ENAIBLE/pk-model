"""Administration route validation shared by simulation APIs."""

from math import isfinite


def validate_route(protocol: str, Ka: float | None) -> str:
    """Normalize a route and validate its absorption parameters."""
    if not isinstance(protocol, str):
        raise TypeError("protocol must be a route string.")
    route = protocol.upper()
    route = {
        "INTRAVENOUS": "IV",
        "SUBCUTANEOUS": "SC",
        "INTRAMUSCULAR": "IM",
        "INTRADERMAL": "ID",
    }.get(route, route)
    if route in {"IM", "ID"}:
        raise NotImplementedError(f"The {route} protocol is not implemented.")
    if route not in {"IV", "SC"}:
        raise ValueError(f"Unknown administration route: {protocol!r}.")
    if route == "SC" and (Ka is None or not isfinite(Ka) or Ka <= 0):
        raise ValueError("SC requires a positive finite absorption rate Ka.")
    return route
