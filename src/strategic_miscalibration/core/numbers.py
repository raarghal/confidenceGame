"""Exact rational coercion.

The theory layer computes in :class:`fractions.Fraction` so that equalities the paper relies on
(a belief sitting *exactly* on a trust boundary, a user *exactly* indifferent) are decided without
tolerances. Inputs arrive as floats from configs and data, so every entry point converts through
:func:`exact`, which reads a float by its shortest decimal representation: ``exact(0.85) == 17/20``,
not the binary expansion of 0.85.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational

__all__ = ["Number", "exact"]

Number = Fraction | int | float | str


def exact(x: Number) -> Fraction:
    """Return ``x`` as a Fraction, reading floats by their shortest decimal form.

    Args:
        x: A Fraction, int, float or decimal string.

    Returns:
        The exact rational value.

    Raises:
        TypeError: For a value with no exact decimal reading (e.g. ``bool`` or ``None``).
    """
    if isinstance(x, bool):
        raise TypeError("refusing to read a bool as a number")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Rational):
        return Fraction(int(x.numerator), int(x.denominator))
    if isinstance(x, float | str):
        return Fraction(repr(x) if isinstance(x, float) else x)
    raise TypeError(f"cannot read {x!r} as an exact number")
