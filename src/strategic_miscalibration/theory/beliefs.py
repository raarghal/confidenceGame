"""The exact Bayes engine: the one place a posterior is computed.

Beliefs are *laws* on the type cells ``(η, ability)`` (see :mod:`.game`). The user conditions on
everything public in a period: every agent's report, her own action, and whichever outcomes the
monitoring reveals. The engine sums over the shared latent difficulty, so reports of transparent
agents act as coupled evidence about the strategic agent's draw without any special casing.

All arithmetic is exact (:class:`~fractions.Fraction`). Functions return *unnormalized* weights where a
caller needs probabilities of events (continuation values weight terminal beliefs by how likely they
are), and :func:`normalize` turns weights into a law.

The user's action is a function of the reports, so its probability is common to every cell and is
left out of the weights: callers that need it multiply it in.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from ..core.numbers import Number, exact
from .game import HONEST, SUBJECT, Cell, Game

__all__ = [
    "Law",
    "Strategy",
    "ability_share",
    "binary_prior",
    "expected_success",
    "honesty",
    "normalize",
    "outcome_events",
    "posterior",
    "prior",
    "report_probability",
    "success_moments",
    "weights",
]

Law = Mapping[Cell, Fraction]


@dataclass(frozen=True)
class Strategy:
    """The strategic type's reporting rule ``σ(message | ability, draw)``.

    Rules are looked up by ``(ability, draw)`` first and then by ``(None, draw)``, so a *common rule*
    (the same for every ability, as in the paper's Definition C.3) is stored once.
    """

    rule: Mapping[tuple[str | None, str], Mapping[str, Fraction]]

    def __post_init__(self) -> None:
        frozen = {}
        for key, dist in self.rule.items():
            d = {m: exact(p) for m, p in dist.items()}
            if sum(d.values()) != 1 or min(d.values()) < 0:
                raise ValueError(f"σ(· | {key}) must be a distribution, got {d}")
            frozen[key] = MappingProxyType(d)
        object.__setattr__(self, "rule", MappingProxyType(frozen))

    def prob(self, ability: str, draw: str, message: str) -> Fraction:
        """``σ(message | ability, draw)``."""
        dist = self.rule.get((ability, draw)) or self.rule[(None, draw)]
        return dist.get(message, Fraction(0))

    @classmethod
    def common(cls, by_draw: Mapping[str, Mapping[str, Number]]) -> Strategy:
        """The same rule for every ability."""
        return cls({(None, draw): {m: exact(p) for m, p in dist.items()} for draw, dist in by_draw.items()})

    @classmethod
    def binary(cls, sigma_plus: Number, sigma_minus: Number) -> Strategy:
        """The paper's ``(σ⁺, σ⁻)``: the probability of reporting ``HIGH`` on an easy and on a hard draw."""
        sp, sm = exact(sigma_plus), exact(sigma_minus)
        return cls.common({"easy": {"HIGH": sp, "LOW": 1 - sp}, "hard": {"HIGH": sm, "LOW": 1 - sm}})

    @classmethod
    def pooling(cls, game: Game, message: str) -> Strategy:
        """Report ``message`` on every draw (e.g. the final-period inflation ``(1, 1)``)."""
        return cls.common({d: {message: Fraction(1)} for d in game.types.draws})


def prior(game: Game, h: Number, ability_probs: Mapping[str, Number]) -> dict[Cell, Fraction]:
    """A product prior: honesty independent of ability."""
    hh = exact(h)
    probs = {a: exact(p) for a, p in ability_probs.items()}
    if set(probs) != set(game.types.abilities) or sum(probs.values()) != 1:
        raise ValueError(f"ability_probs must be a distribution over {game.types.abilities}")
    return {(eta, a): (hh if eta == HONEST else 1 - hh) * probs[a] for eta, a in game.cells}


def binary_prior(game: Game, h: Number, mu: Number) -> dict[Cell, Fraction]:
    """The paper's ``(h, μ)`` prior, ``μ`` being the probability of the first (high) ability."""
    a_hi, a_lo = game.types.abilities[:2]
    m = exact(mu)
    probs = {a: Fraction(0) for a in game.types.abilities}
    probs.update({a_hi: m, a_lo: 1 - m})
    return prior(game, h, probs)


def normalize(w: Law) -> dict[Cell, Fraction] | None:
    """Weights -> law; ``None`` if the weights have no mass (a zero-probability event)."""
    total = sum(w.values())
    return None if total == 0 else {c: v / total for c, v in w.items()}


def honesty(law: Law) -> Fraction:
    """Marginal belief that the agent is honest, ``h``."""
    return sum((v for (eta, _), v in law.items() if eta == HONEST), Fraction(0))


def ability_share(law: Law, ability: str) -> Fraction:
    """Marginal belief in one ability (``μ`` for the high ability)."""
    return sum((v for (_, a), v in law.items() if a == ability), Fraction(0))


def _abilities(game: Game, cell: Cell) -> dict[str, str]:
    """Each agent's ability in a world where the strategic agent's type is ``cell``."""
    return {SUBJECT: cell[1], **{t.name: t.ability for t in game.roster.transparent}}


def _report_prob(game: Game, strategy: Strategy, honest: bool, ability: str, draw: str, message: str) -> Fraction:
    if honest:
        return Fraction(int(game.reports.honest(draw, game.types.rho(draw, ability)) == message))
    return strategy.prob(ability, draw, message)


def _worlds(
    game: Game, strategy: Strategy, cell: Cell, reports: Mapping[str, str]
) -> Iterator[tuple[Fraction, dict[str, str], dict[str, str]]]:
    """``(P(ω-cell and reports | type), draws, abilities)`` over the latent difficulty partition."""
    abilities = _abilities(game, cell)
    for p_omega, by_ability in game.types.latent_cells(tuple(dict.fromkeys(abilities.values()))):
        draws = {agent: by_ability[a] for agent, a in abilities.items()}
        lik = p_omega
        for agent, message in reports.items():
            honest = agent != SUBJECT or cell[0] == HONEST
            lik *= _report_prob(game, strategy, honest, abilities[agent], draws[agent], message)
            if lik == 0:
                break
        if lik:
            yield lik, draws, abilities


def weights(
    game: Game,
    law: Law,
    strategy: Strategy,
    reports: Mapping[str, str],
    action: str | None = None,
    outcomes: Mapping[str, bool] | None = None,
) -> dict[Cell, Fraction]:
    """Unnormalized posterior weights after one period's public events.

    Args:
        game: The game.
        law: The belief before the period.
        strategy: The strategic type's rule the user conjectures.
        reports: The reports observed, by agent. Agents left out are not conditioned on.
        action: The user's action. When given, the weights include the probability that exactly the
            outcomes in ``outcomes`` were revealed, as the monitoring dictates.
        outcomes: Revealed outcomes by agent (``True`` = success). Ignored without ``action``.

    Returns:
        ``P(type, events)`` for every cell.
    """
    outcomes = outcomes or {}
    out: dict[Cell, Fraction] = {}
    for cell, w in law.items():
        total = Fraction(0)
        if w:
            for lik, draws, abilities in _worlds(game, strategy, cell, reports):
                if action is not None:
                    for agent in game.roster.agents:
                        r = game.monitoring.reveal_prob(action, agent)
                        if agent in outcomes:
                            rho = game.types.rho(draws[agent], abilities[agent])
                            lik *= r * (rho if outcomes[agent] else 1 - rho)
                        else:
                            lik *= 1 - r
                total += lik
        out[cell] = w * total
    return out


def posterior(
    game: Game,
    law: Law,
    strategy: Strategy,
    reports: Mapping[str, str],
    action: str | None = None,
    outcomes: Mapping[str, bool] | None = None,
) -> dict[Cell, Fraction] | None:
    """Normalized :func:`weights`; ``None`` after a zero-probability event."""
    return normalize(weights(game, law, strategy, reports, action, outcomes))


def report_probability(game: Game, law: Law, strategy: Strategy, reports: Mapping[str, str]) -> Fraction:
    """Probability of observing ``reports``."""
    return sum(weights(game, law, strategy, reports).values(), Fraction(0))


def success_moments(
    game: Game, law: Law, strategy: Strategy, reports: Mapping[str, str], agent: str = SUBJECT
) -> tuple[Fraction, Fraction]:
    """``(P(reports and agent succeeds), P(reports))``; affine in ``strategy`` when one rule entry varies."""
    num = den = Fraction(0)
    for cell, w in law.items():
        if not w:
            continue
        for lik, draws, abilities in _worlds(game, strategy, cell, reports):
            den += w * lik
            num += w * lik * game.types.rho(draws[agent], abilities[agent])
    return num, den


def expected_success(
    game: Game, law: Law, strategy: Strategy, reports: Mapping[str, str], agent: str = SUBJECT
) -> Fraction | None:
    """``E[agent succeeds | reports]`` (the paper's ``ρ̃``); ``None`` if ``reports`` has probability zero."""
    num, den = success_moments(game, law, strategy, reports, agent)
    return None if den == 0 else num / den


def outcome_events(game: Game, action: str) -> list[dict[str, bool]]:
    """Every pattern of revealed outcomes that can follow ``action`` (``{}`` = nothing revealed)."""
    patterns: list[dict[str, bool]] = [{}]
    for agent in game.roster.agents:
        r = game.monitoring.reveal_prob(action, agent)
        if r == 0:
            continue
        revealed = [{**p, agent: ok} for p in patterns for ok in (True, False)]
        patterns = revealed if r == 1 else patterns + revealed
    return patterns
