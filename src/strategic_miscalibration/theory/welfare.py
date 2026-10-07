"""Two-period welfare: what a reporting rule costs the user (Section 6, Appendix P).

**Timeline.** Period 1: the strategic types play ``rule1`` (in the paper, the rule measured at the
state), honest types report truthfully. The user decides with her belief and a conjecture about the
strategic rule, then updates by Bayes on the joint law: on the report, and on the outcome if the
monitoring reveals it. Period 2: the strategic types play ``rule2`` (by default the final-period
inflation, pooling on the top message: an assumption the experiments cannot test, since they elicit
one first-period action per state) and the user decides again under her second conjecture.

**Who is who.** The user's *belief* sets her decisions; the *population* is who she actually faces
and sets the outcomes. The paper's four users differ only in belief and conjectures (:func:`user`):

=============  ======================  =======================  ======================
user           belief                  period-1 conjecture      period-2 conjecture
=============  ======================  =======================  ======================
``naive``      the state ``(h, μ)``    truthful                 truthful
``soph``       the state ``(h, μ)``    the actual rule          full inflation
``informed``   the population          the actual rule          full inflation
benchmark      the state ``(h, μ)``    truthful                 truthful, against an all-honest population
=============  ======================  =======================  ======================

**Accounting.** The user is myopic but lives through both periods, so her welfare is the undiscounted
sum. The agent weights ``δ`` on period 1 and ``1 − δ`` on period 2, as its prompt states. Everything
is an exact expectation over types, draws, reports and revealed outcomes. Nothing is simulated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .beliefs import Law, Strategy, posterior
from .game import HONEST, SELF, SUBJECT, Game
from .users import OracleUser

__all__ = ["Payoffs2", "UserModel", "benchmark", "two_periods", "user"]

UserKind = Literal["naive", "soph", "informed"]


@dataclass(frozen=True)
class UserModel:
    """A user: her belief and her conjecture about the strategic rule in each period."""

    belief: Law
    conjecture1: Strategy
    conjecture2: Strategy


@dataclass(frozen=True)
class Payoffs2:
    """Expected stage payoffs of the user and the agent in each period."""

    user1: Fraction
    user2: Fraction
    agent1: Fraction
    agent2: Fraction

    @property
    def user(self) -> Fraction:
        """The user's two-period welfare: the undiscounted sum."""
        return self.user1 + self.user2

    def agent(self, delta: Fraction) -> Fraction:
        """The agent's value, ``δ`` on period 1 and ``1 − δ`` on period 2."""
        return delta * self.agent1 + (1 - delta) * self.agent2


def truthful(game: Game) -> Strategy:
    """The strategic type reporting as an honest one would."""
    return Strategy.common(
        {d: {game.reports.honest(d, game.types.rho(d, game.types.abilities[0])): 1} for d in game.types.draws}
    )


def user(game: Game, kind: UserKind, state: Law, population: Law, rule: Strategy) -> UserModel:
    """One of the paper's three users at a belief state, against a population playing ``rule``."""
    inflate = Strategy.pooling(game, game.reports.top)
    if kind == "naive":
        return UserModel(state, truthful(game), truthful(game))
    if kind == "soph":
        return UserModel(state, rule, inflate)
    if kind == "informed":
        return UserModel(population, rule, inflate)
    raise ValueError(f"unknown user {kind!r}")


def _all_honest(population: Law) -> dict:
    """The same abilities, every type honest."""
    out = dict.fromkeys(population, Fraction(0))
    for (_, a), w in population.items():
        out[(HONEST, a)] += w
    return out


def benchmark(game: Game, state: Law, population: Law) -> Payoffs2:
    """The honest benchmark ``W_honest``: the same abilities, every type truthful, a naive user at ``state``."""
    rule = truthful(game)
    return two_periods(game, UserModel(state, rule, rule), _all_honest(population), rule, rule)


def _delegated(game: Game, model: Law, conjecture: Strategy, message: str) -> bool:
    return OracleUser(game, conjecture).decide(model, {SUBJECT: message}).action == SUBJECT


def _message_probs(game: Game, cell: tuple[int, str], draw: str, rule: Strategy) -> Mapping[str, Fraction]:
    eta, ability = cell
    if eta == HONEST:
        return {game.reports.honest(draw, game.types.rho(draw, ability)): Fraction(1)}
    return {m: rule.prob(ability, draw, m) for m in game.reports.messages}


def _revealed(game: Game, action: str, success_prob: Fraction) -> list[tuple[Fraction, dict[str, bool]]]:
    """``(probability, revealed outcomes)`` after ``action`` for an agent with this success probability."""
    r = game.monitoring.reveal_prob(action, SUBJECT)
    out = [(r * success_prob, {SUBJECT: True}), (r * (1 - success_prob), {SUBJECT: False})] if r else []
    if r < 1:
        out.append((1 - r, {}))
    return out


def _stage(game: Game, delegate: bool, success_prob: Fraction, audited: bool) -> tuple[Fraction, Fraction]:
    """(user, agent) expected stage payoff given the decision and the agent's success probability."""
    pay = game.payoffs
    if delegate:
        return pay.delegate_value(success_prob), pay.cost
    audit_cost = game.monitoring.audit_cost if audited else Fraction(0)
    return pay.self_value() - audit_cost, Fraction(0)


def two_periods(
    game: Game, model: UserModel, population: Law, rule1: Strategy, rule2: Strategy | None = None
) -> Payoffs2:
    """Expected payoffs over two periods.

    Args:
        game: A monopoly game.
        model: The user (belief and conjectures).
        population: The true distribution of agent types.
        rule1: The strategic types' period-1 rule.
        rule2: Their period-2 rule; default full inflation.

    Returns:
        Expected stage payoffs of both players in both periods.
    """
    if not game.roster.is_monopoly:
        raise NotImplementedError("two-period welfare is implemented for the monopoly game")
    rule2 = rule2 or Strategy.pooling(game, game.reports.top)
    d1 = {m: _delegated(game, model.belief, model.conjecture1, m) for m in game.reports.messages}
    u1 = u2 = a1 = a2 = Fraction(0)
    for cell, w in population.items():
        if not w:
            continue
        ability = cell[1]
        for draw in game.types.draws:
            p_draw = game.types.p_draw(draw, ability)
            succ = game.types.rho(draw, ability)
            for message, p_msg in _message_probs(game, cell, draw, rule1).items():
                p = w * p_draw * p_msg
                if not p:
                    continue
                action = SUBJECT if d1[message] else SELF
                for p_obs, outcomes in _revealed(game, action, succ):
                    weight = p * p_obs
                    if not weight:
                        continue
                    du, da = _stage(game, d1[message], succ, audited=bool(outcomes) and action == SELF)
                    u1, a1 = u1 + weight * du, a1 + weight * da
                    post = posterior(game, model.belief, model.conjecture1, {SUBJECT: message}, action, outcomes)
                    belief2 = model.belief if post is None else post
                    du2, da2 = _period2(game, cell, belief2, model.conjecture2, rule2)
                    u2, a2 = u2 + weight * du2, a2 + weight * da2
    return Payoffs2(u1, u2, a1, a2)


def _period2(
    game: Game, cell: tuple[int, str], belief: Law, conjecture: Strategy, rule: Strategy
) -> tuple[Fraction, Fraction]:
    """The final period's expected stage payoffs against one known type."""
    d2 = {m: _delegated(game, belief, conjecture, m) for m in game.reports.messages}
    u = a = Fraction(0)
    ability = cell[1]
    for draw in game.types.draws:
        p_draw = game.types.p_draw(draw, ability)
        succ = game.types.rho(draw, ability)
        for message, p_msg in _message_probs(game, cell, draw, rule).items():
            p = p_draw * p_msg
            if not p:
                continue
            audited = not d2[message] and game.monitoring.audit_prob > 0
            du, da = _stage(game, d2[message], succ, audited=False)
            if audited:
                du -= game.monitoring.audit_prob * game.monitoring.audit_cost
            u, a = u + p * du, a + p * da
    return u, a
