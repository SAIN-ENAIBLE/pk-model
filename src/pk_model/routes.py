"""Compatibility access to route validation owned by Model."""

from .model import Model


def validate_route(protocol: str, Ka: float | None) -> str:
    """Normalise a route using the model's shared validation rules."""
    return Model.validate_route(protocol, Ka)
