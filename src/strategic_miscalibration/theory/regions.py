"""The paper's first-period predictions on the experiment grid: trust regions and equilibrium bluffing rates.

These are what Section 5 scores the measured strategies against. Earlier analysis code hardcoded them
(``R_SET``, ``F_SET``, ``X0_F``); here they are computed from the solver at the game's primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ..core.numbers import Number
from .beliefs import Strategy, binary_prior, weights
from .game import SUBJECT, Game
from .solvers import solve
from .terminal import is_trusted

__all__ = ["Region", "StatePrediction", "predict", "region", "standard_user_sigma_minus"]

Region = Literal["R", "F", "D", "out"]


def region(game: Game, h: Number, mu: Number) -> Region:
    """Where a product prior sits relative to the first-period trust regions.

    ``out``: the prior is not trusted even before any report (outside ``τ_C``), so the theory makes no
    first-period prediction. Otherwise, under the inflation rule ``(1, 1)``:
    ``R`` (robust) — trust survives a delegated *failure* after a high report;
    ``F`` (fragile) — trust survives a success but not a failure;
    ``D`` — trust survives neither.
    """
    law = binary_prior(game, h, mu)
    if not is_trusted(game, law):
        return "out"
    inflate = Strategy.binary(1, 1)

    def trusted_after(success: bool) -> bool:
        return is_trusted(game, weights(game, law, inflate, {SUBJECT: "HIGH"}, SUBJECT, {SUBJECT: success}))

    if trusted_after(False):
        return "R"
    if trusted_after(True):
        return "F"
    return "D"


def standard_user_sigma_minus(game: Game, h: Number, mu: Number, delta: Number) -> Fraction | None:
    """``σ⁻`` of the payoff-relevant equilibrium with the standard user ``(d⁺, d⁻) = (1, 0)`` (Thm 4.3).

    Unique inside ``τ_C`` at generic parameters; ``None`` where no such equilibrium exists.
    """
    standard = [p for p in solve(game, h, mu, delta) if p.payoff_relevant and p.user == (1, 0)]
    if not standard:
        return None
    rates = {p.sigma_minus for p in standard}
    if len(rates) > 1:
        raise ValueError(f"standard-user equilibrium not unique at {(h, mu, delta)}: {sorted(rates)}")
    return rates.pop()


@dataclass(frozen=True)
class StatePrediction:
    h: Fraction
    mu: Fraction
    delta: Fraction
    region: Region
    sigma_minus: Fraction | None  # None outside τ_C or where the standard-user equilibrium does not exist


def predict(game: Game, h: Number, mu: Number, delta: Number) -> StatePrediction:
    """Region and equilibrium bluffing rate at one state."""
    from ..core.numbers import exact

    reg = region(game, h, mu)
    rate = None if reg == "out" else standard_user_sigma_minus(game, h, mu, delta)
    return StatePrediction(exact(h), exact(mu), exact(delta), reg, rate)
