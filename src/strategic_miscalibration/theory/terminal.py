"""The final period: the trust test, the trust index, and the trust regions (Thm 4.1).

In the last period there is no future to protect, so the strategic type's best response to any user
who delegates its top message is to send the top message on every draw. The agent-optimal final-period
equilibrium therefore pays the agent the fee ``c`` exactly when the user still delegates under that
pooling, i.e. when ``E[success | top message] >= ρ*`` with strategic types pooling on top. At equality
the user is indifferent and may mix, which is what makes boundary equilibria possible in the first
period (:mod:`.solvers.binary_t2`).

This test is computed by the Bayes engine and so holds for any type space (graded success included).
It assumes a monopoly and a binary report space; other games need their own final-period solver.

For the paper's binary type space the test has a closed form in the *trust index*
``Ψ̂(π) = P(easy) / P(strategic ∧ hard)``: trusted iff ``Ψ̂ ≥ Ψ* = (ρ* − ρ⁻) / (ρ⁺ − ρ*)``.
"""

from __future__ import annotations

from fractions import Fraction

from .beliefs import Law, Strategy, success_moments
from .game import STRATEGIC, SUBJECT, Game

__all__ = ["final_success_gap", "is_trusted", "psi_star", "terminal_value", "trust_index"]


def _require_terminal_scope(game: Game) -> None:
    if not game.roster.is_monopoly or not game.reports.is_binary:
        raise NotImplementedError(
            "the pooling final-period test covers monopoly games with a binary report space; "
            f"got roster={game.roster.agents}, reports={game.reports.messages}"
        )


def final_success_gap(game: Game, law: Law) -> Fraction | None:
    """``P(top) · (E[success | top] − ρ*)`` under final-period pooling; ``None`` if no one sends ``top``.

    The sign decides the trust test, and the expression is affine in the law, which the boundary
    equilibria exploit.
    """
    _require_terminal_scope(game)
    num, den = success_moments(game, law, Strategy.pooling(game, game.reports.top), {SUBJECT: game.reports.top})
    return None if den == 0 else num - game.rho_star * den


def is_trusted(game: Game, law: Law) -> bool:
    """Would the final-period user delegate the top message when strategic types pool on it?"""
    gap = final_success_gap(game, law)
    return gap is None or gap >= 0


def terminal_value(game: Game, law: Law) -> Fraction:
    """The strategic agent's final-period value under agent-optimal selection: ``c`` if trusted, else 0."""
    return game.cost if is_trusted(game, law) else Fraction(0)


def psi_star(game: Game) -> Fraction:
    """``Ψ* = (ρ* − ρ⁻) / (ρ⁺ − ρ*)`` for the paper's binary type space."""
    _require_binary_types(game)
    a = game.types.abilities[0]
    rp, rm = game.types.rho("easy", a), game.types.rho("hard", a)
    return (game.rho_star - rm) / (rp - game.rho_star)


def trust_index(game: Game, law: Law) -> Fraction | None:
    """``Ψ̂(π) = P(easy) / P(strategic ∧ hard)``; ``None`` stands for ``+∞``."""
    _require_binary_types(game)
    easy = sum((v * game.types.p_draw("easy", a) for (_, a), v in law.items()), Fraction(0))
    bluffable = sum((v * game.types.p_draw("hard", a) for (eta, a), v in law.items() if eta == STRATEGIC), Fraction(0))
    return None if bluffable == 0 else easy / bluffable


def _require_binary_types(game: Game) -> None:
    if game.types.draws != ("easy", "hard") or game.types.success_depends_on_ability:
        raise NotImplementedError("the trust index is the closed form for the paper's binary, ungraded type space")
