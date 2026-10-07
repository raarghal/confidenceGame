"""The Bayes-rational user.

The user is myopic: each period she takes the action with the highest expected stage payoff given
everything public, completing the task herself (``r − e``) or delegating to an agent ``i``
(``r · E[success_i | reports] − c``). Her inference uses a *conjecture* about the strategic type's
reporting rule; with the conjecture equal to the agent's actual rule this is the equilibrium user.

Ties between delegating and not are broken toward delegating, and ties between agents toward a
transparent (verifiable) agent (a verified high signal wins). A report the conjecture assigns probability
zero leaves her indifferent, so it is delegated. Equilibrium analysis lets the user mix at
indifference; that is the solvers' business, not this class's.

Any object with ``decide`` and ``update`` of these signatures can stand in for this user (an LLM user, a
scripted user): see :class:`User`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from .beliefs import Law, Strategy, expected_success, posterior
from .game import SELF, SUBJECT, Game

__all__ = ["Decision", "OracleUser", "User"]


@dataclass(frozen=True)
class Decision:
    """A user's choice and the expected stage payoff of every action she weighed."""

    action: str
    payoffs: Mapping[str, Fraction]


class User(Protocol):
    """What the engine needs from any user."""

    def decide(self, law: Law, reports: Mapping[str, str]) -> Decision: ...

    def update(
        self, law: Law, reports: Mapping[str, str], action: str, outcomes: Mapping[str, bool]
    ) -> Law | None: ...


@dataclass(frozen=True)
class OracleUser:
    """Exact Bayes-rational, myopic user with a conjecture about the strategic type's rule."""

    game: Game
    conjecture: Strategy

    def decide(self, law: Law, reports: Mapping[str, str]) -> Decision:
        """The payoff-maximizing action after observing ``reports``."""
        pay = self.game.payoffs
        payoffs: dict[str, Fraction] = {SELF: pay.self_value()}
        for agent in self.game.roster.agents:
            if agent not in reports:
                continue
            succ = expected_success(self.game, law, self.conjecture, reports, agent)
            # A report the conjecture deems impossible leaves her indifferent (success read as ρ*).
            payoffs[agent] = pay.delegate_value(self.game.rho_star if succ is None else succ)
        # Rank: payoff, then delegation over self-completion, then transparent agents over the subject.
        order = {SELF: 0, SUBJECT: 1}
        return Decision(max(payoffs, key=lambda a: (payoffs[a], order.get(a, 2))), payoffs)

    def update(self, law: Law, reports: Mapping[str, str], action: str, outcomes: Mapping[str, bool]) -> Law | None:
        """The posterior after the period's reports, the action taken, and the outcomes revealed."""
        return posterior(self.game, law, self.conjecture, reports, action, outcomes)
