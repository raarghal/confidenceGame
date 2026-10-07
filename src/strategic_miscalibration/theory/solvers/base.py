"""Equilibrium solvers, registered by the games they cover.

A solver states which games it applies to (:meth:`Solver.unsupported` returns a reason when it does
not) and enumerates first-period equilibria at a state. :func:`solve` dispatches to the first solver
that covers the game, so a solver for a new generalization (graded success, more messages, a
competitor) is a new module registered here; callers never change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from ...core.numbers import Number
from ...core.registry import Registry
from ..game import Game

__all__ = ["SOLVERS", "NoSolverError", "Profile", "Solver", "solve"]


@dataclass
class Profile:
    """One first-period equilibrium of a two-message game.

    Attributes:
        d_plus, d_minus: Probability the user delegates after a high / low report.
        sigma_plus, sigma_minus: Probability the strategic type reports high on an easy / hard draw.
        family: How it was found (``corner``, ``edge:<name>``, ``boundary:<events>``).
        payoff_relevant: Whether the report changes the agent's payoff at either draw.
        delta_plus, delta_minus: The agent's gain from reporting high rather than low, by draw.
        agent_value: The agent's ex-ante value, prior-weighted over ability.
        values: Continuation value of every terminal belief, keyed ``event + report`` (e.g. ``"fail+"``).
        terminal_mix: For boundary equilibria, the events sharing the boundary belief and the user's
            delegation probability there.
        flags: Non-generic coincidences met along the way.
    """

    d_plus: Fraction
    d_minus: Fraction
    sigma_plus: Fraction
    sigma_minus: Fraction
    family: str
    payoff_relevant: bool
    delta_plus: Fraction
    delta_minus: Fraction
    agent_value: Fraction
    values: dict[str, Fraction]
    terminal_mix: tuple[str, Fraction] | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def user(self) -> tuple[Fraction, Fraction]:
        return self.d_plus, self.d_minus

    @property
    def sigma(self) -> tuple[Fraction, Fraction]:
        return self.sigma_plus, self.sigma_minus

    def as_floats(self) -> tuple[float, float, float, float]:
        return float(self.d_plus), float(self.d_minus), float(self.sigma_plus), float(self.sigma_minus)


class Solver(Protocol):
    """Enumerates first-period equilibria for the games it covers."""

    def unsupported(self, game: Game) -> str | None:
        """``None`` if this solver covers ``game``, else the reason it does not."""
        ...

    def equilibria(self, game: Game, h: Number, mu: Number, delta: Number) -> list[Profile]:
        """Every equilibrium found at the product prior ``(h, μ)`` and discount ``δ``."""
        ...


SOLVERS: Registry[Solver] = Registry("solver")


class NoSolverError(NotImplementedError):
    """No registered solver covers the game."""


def solve(game: Game, h: Number, mu: Number, delta: Number) -> list[Profile]:
    """Equilibria at a state, from the first registered solver that covers ``game``.

    Raises:
        NoSolverError: Listing why each registered solver declined.
    """
    reasons = []
    for name, solver in SOLVERS.items():
        why = solver.unsupported(game)
        if why is None:
            return solver.equilibria(game, h, mu, delta)
        reasons.append(f"{name}: {why}")
    raise NoSolverError("no solver covers this game:\n  " + "\n  ".join(reasons))
