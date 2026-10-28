from collections.abc import Callable


def linear(cl: float) -> Callable[[float, float], float]:
    """Linear clearance: returns clear(t, conc) -> amount / time."""

    def clear(t: float, conc: float) -> float:
        # TODO: CL is volume / time and conc is amount / volume.
        #       Which combination gives amount / time? (Update test_clear.py first.)
        raise NotImplementedError

    return clear
