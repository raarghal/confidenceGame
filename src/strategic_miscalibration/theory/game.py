"""The Confidence Game as a set of orthogonal axes.

A :class:`Game` is the product of independent choices, each a small frozen value:

=================  ==========================  ==================================================
axis               type                        the paper's value
=================  ==========================  ==================================================
payoffs            :class:`Payoffs`            ``r = 1``, ``c = 0.1``, ``e = 0.5`` (so ``ρ* = 0.6``)
task and types     :class:`TypeSpace`          abilities ``θ_H = 0.8``, ``θ_L = 0.2`` (the chance of
                                               an easy draw); success ``ρ⁺ = 0.85``, ``ρ⁻ = 0.15``
reports            :class:`ReportSpace`        binary: ``LOW`` / ``HIGH``
monitoring         :class:`Monitoring`         endogenous: an outcome is seen only if delegated
market             :class:`Roster`             monopoly: the strategic agent alone
horizon            ``Game.horizon``            two periods
=================  ==========================  ==================================================

Each generalization the project anticipates is a different value on one axis, never a new code path:
graded success is a :class:`TypeSpace` whose success depends on ability; quantized confidence is a
:class:`ReportSpace` with more messages; auditing is a :class:`Monitoring` with ``audit_prob > 0``; an
honest competitor is a :class:`Roster` with a :class:`TransparentAgent`. Everything downstream (the
Bayes engine in :mod:`.beliefs`, the oracle user, welfare) is written against these types, and solvers
declare which games they cover (:mod:`.solvers`).

**Types.** The agent's private type is a *cell* ``(η, a)``: honesty ``η`` (``HONEST = 1`` or
``STRATEGIC = 0``) and ability ``a``. The user's belief is a law on cells — the *joint* law, which does
not factor into an honesty and an ability marginal once a non-honest report has been observed.

**Draws and coupling.** Every period a latent task difficulty ``ω ~ U[0, 1]`` is drawn once and shared
by every agent. An agent of ability ``a`` gets the ``k``-th draw (``draws[k]``, ordered best first) when
``ω`` falls in the ``k``-th interval of ``a``'s cumulative draw distribution. Draws are therefore
comonotone across agents: a weaker agent's easy draw implies a stronger agent's. With one agent this is
just ``P(draw | a)``; with a competitor it is the draw–audit coupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from types import MappingProxyType

from ..core.numbers import Number, exact

__all__ = [
    "ENDOGENOUS",
    "EXOGENOUS",
    "HONEST",
    "MONOPOLY",
    "STRATEGIC",
    "SUBJECT",
    "Cell",
    "Game",
    "Monitoring",
    "Payoffs",
    "ReportSpace",
    "Roster",
    "TransparentAgent",
    "TypeSpace",
    "kappa",
    "paper_game",
]

HONEST, STRATEGIC = 1, 0
SUBJECT = "S"  # the strategic agent under study; the user's "delegate to the agent" action
SELF = "self"  # the user's "complete the task herself" action

Cell = tuple[int, str]  # (honesty η, ability name)


def _frozen(mapping: Mapping) -> Mapping:
    return MappingProxyType(dict(mapping))


def kappa(delta: Number) -> Fraction:
    """Myopia ratio ``κ = δ / (1 − δ)``: the weight on today's fee relative to the future.

    In the prompts and the paper, ``δ`` weights the *current* round, so large ``δ`` is myopic.
    """
    d = exact(delta)
    return d / (1 - d)


def delta_of_kappa(k: Number) -> Fraction:
    """Inverse of :func:`kappa`."""
    kk = exact(k)
    return kk / (1 + kk)


@dataclass(frozen=True)
class Payoffs:
    """Stage payoffs.

    The user earns ``r`` from a completed task. Self-completion costs effort ``e`` and always succeeds;
    delegation costs the fee ``c``, which is the agent's revenue, and succeeds with the agent's success
    probability. A myopic user therefore delegates iff the expected success is at least
    ``ρ* = 1 − (e − c) / r``.
    """

    reward: Fraction = Fraction(1)
    cost: Fraction = Fraction(1, 10)
    effort: Fraction = Fraction(1, 2)

    def __post_init__(self) -> None:
        for name in ("reward", "cost", "effort"):
            object.__setattr__(self, name, exact(getattr(self, name)))
        if not 0 < self.cost < self.effort <= self.reward:
            raise ValueError("need 0 < c < e <= r so that delegation is sometimes but not always worthwhile")

    @property
    def rho_star(self) -> Fraction:
        """Delegation threshold ``ρ* = 1 − (e − c) / r``."""
        return 1 - (self.effort - self.cost) / self.reward

    def self_value(self) -> Fraction:
        """User payoff from completing the task herself."""
        return self.reward - self.effort

    def delegate_value(self, success: Fraction) -> Fraction:
        """User payoff from delegating to an agent with expected success ``success``."""
        return self.reward * success - self.cost


@dataclass(frozen=True)
class TypeSpace:
    """Abilities, draws, and success probabilities.

    Attributes:
        draws: Draw labels ordered from best to worst, e.g. ``("easy", "hard")``.
        abilities: Ability labels, e.g. ``("H", "L")``. Their order fixes the meaning of a scalar
            ``μ`` (the probability of the first ability) in two-ability games.
        draw_probs: ``ability -> P(draw | ability)``, aligned with ``draws``.
        success: ``(draw, ability) -> success probability``.
    """

    draws: tuple[str, ...]
    abilities: tuple[str, ...]
    draw_probs: Mapping[str, tuple[Fraction, ...]]
    success: Mapping[tuple[str, str], Fraction]
    _latent: dict[tuple[str, ...], list[tuple[Fraction, dict[str, str]]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        probs = {a: tuple(exact(p) for p in ps) for a, ps in self.draw_probs.items()}
        succ = {k: exact(v) for k, v in self.success.items()}
        object.__setattr__(self, "draw_probs", _frozen(probs))
        object.__setattr__(self, "success", _frozen(succ))
        if set(probs) != set(self.abilities):
            raise ValueError("draw_probs must give one distribution per ability")
        for a, ps in probs.items():
            if len(ps) != len(self.draws) or sum(ps) != 1 or min(ps) < 0:
                raise ValueError(f"P(draw | {a}) must be a distribution over {self.draws}")
        missing = {(d, a) for d in self.draws for a in self.abilities} - set(succ)
        if missing:
            raise ValueError(f"success probability missing for {sorted(missing)}")

    @classmethod
    def binary(cls, theta_H: Number, theta_L: Number, rho_plus: Number, rho_minus: Number) -> TypeSpace:
        """The paper's type space: ability is the chance of an easy draw; success depends only on the draw."""
        tH, tL = exact(theta_H), exact(theta_L)
        rp, rm = exact(rho_plus), exact(rho_minus)
        return cls(
            draws=("easy", "hard"),
            abilities=("H", "L"),
            draw_probs={"H": (tH, 1 - tH), "L": (tL, 1 - tL)},
            success={("easy", "H"): rp, ("easy", "L"): rp, ("hard", "H"): rm, ("hard", "L"): rm},
        )

    @classmethod
    def graded(
        cls, theta_H: Number, theta_L: Number, rho_plus_H: Number, rho_plus_L: Number, rho_minus: Number
    ) -> TypeSpace:
        """Graded success: on an easy draw the high-ability type succeeds more often (``ρ⁺_L < ρ⁺_H``)."""
        base = cls.binary(theta_H, theta_L, rho_plus_H, rho_minus)
        success = dict(base.success)
        success[("easy", "L")] = exact(rho_plus_L)
        return cls(base.draws, base.abilities, base.draw_probs, success)

    def with_ability(self, name: str, draw_probs: tuple[Number, ...], success: Mapping[str, Number]) -> TypeSpace:
        """A copy with one more ability, e.g. a competitor's ability that the subject never has."""
        if name in self.abilities:
            raise ValueError(f"ability {name!r} already exists")
        probs = dict(self.draw_probs)
        probs[name] = tuple(exact(p) for p in draw_probs)
        succ = dict(self.success)
        succ.update({(d, name): exact(success[d]) for d in self.draws})
        return TypeSpace(self.draws, (*self.abilities, name), probs, succ)

    def rho(self, draw: str, ability: str) -> Fraction:
        """Success probability on ``draw`` for ``ability``."""
        return self.success[(draw, ability)]

    def p_draw(self, draw: str, ability: str) -> Fraction:
        """``P(draw | ability)``."""
        return self.draw_probs[ability][self.draws.index(draw)]

    @property
    def success_depends_on_ability(self) -> bool:
        """True for graded success, False for the paper's model."""
        return any(len({self.rho(d, a) for a in self.abilities}) > 1 for d in self.draws)

    def latent_cells(self, abilities: tuple[str, ...]) -> list[tuple[Fraction, dict[str, str]]]:
        """Partition the shared latent difficulty ``ω`` for the given abilities.

        Returns:
            ``(probability, {ability: draw})`` for each interval of ``ω`` on which every listed
            ability's draw is constant. Probabilities sum to one. Cached: callers must not mutate it.
        """
        if abilities not in self._latent:
            self._latent[abilities] = self._partition(abilities)
        cells: list[tuple[Fraction, dict[str, str]]] = self._latent[abilities]
        return cells

    def _partition(self, abilities: tuple[str, ...]) -> list[tuple[Fraction, dict[str, str]]]:
        cuts = {Fraction(0), Fraction(1)}
        for a in abilities:
            acc = Fraction(0)
            for p in self.draw_probs[a]:
                acc += p
                cuts.add(acc)
        points = sorted(cuts)
        cells = []
        for lo, hi in pairwise(points):
            if hi == lo:
                continue
            mid = (lo + hi) / 2
            cells.append((hi - lo, {a: self._draw_at(a, mid) for a in abilities}))
        return cells

    def _draw_at(self, ability: str, omega: Fraction) -> str:
        acc = Fraction(0)
        for draw, p in zip(self.draws, self.draw_probs[ability], strict=True):
            acc += p
            if omega < acc:
                return draw
        return self.draws[-1]


@dataclass(frozen=True)
class ReportSpace:
    """The messages an agent may send.

    Attributes:
        messages: Message labels ordered from lowest to highest confidence.
        values: The confidence each message states, aligned with ``messages``.
        honest_by_draw: How an honest agent reports, by draw. When absent, an honest agent reports the
            message whose value is nearest its success probability (ties go to the higher message).
    """

    messages: tuple[str, ...]
    values: tuple[Fraction, ...]
    honest_by_draw: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(exact(v) for v in self.values))
        if len(self.values) != len(self.messages) or list(self.values) != sorted(self.values):
            raise ValueError("values must be aligned with messages and increasing")
        if self.honest_by_draw is not None:
            object.__setattr__(self, "honest_by_draw", _frozen(self.honest_by_draw))

    @classmethod
    def binary(cls, low: Number = 0, high: Number = 1) -> ReportSpace:
        """``LOW`` / ``HIGH``; an honest agent reports ``HIGH`` exactly on an easy draw."""
        return cls(("LOW", "HIGH"), (exact(low), exact(high)), {"easy": "HIGH", "hard": "LOW"})

    @classmethod
    def grid(cls, k: int) -> ReportSpace:
        """``k`` evenly spaced confidence levels on ``[0, 1]``; an honest agent reports the nearest one."""
        if k < 2:
            raise ValueError("a report space needs at least two messages")
        values = tuple(Fraction(i, k - 1) for i in range(k))
        return cls(tuple(str(float(v)) for v in values), values)

    @property
    def top(self) -> str:
        """The highest-confidence message."""
        return self.messages[-1]

    @property
    def is_binary(self) -> bool:
        return len(self.messages) == 2

    def honest(self, draw: str, success: Fraction) -> str:
        """The message an honest agent sends on ``draw`` with success probability ``success``."""
        if self.honest_by_draw is not None:
            return self.honest_by_draw[draw]
        best = min(range(len(self.values)), key=lambda i: (abs(self.values[i] - success), -i))
        return self.messages[best]


@dataclass(frozen=True)
class Monitoring:
    """What the user learns about outcomes.

    The outcome of a delegated task is always observed. ``audit_prob`` is the chance that the strategic
    agent's would-be outcome is revealed when the user does *not* delegate to it, at a cost ``audit_cost``
    to the user. ``audit_prob = 0`` is the paper's endogenous monitoring; ``audit_prob = 1`` is exogenous
    monitoring.
    """

    audit_prob: Fraction = Fraction(0)
    audit_cost: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_prob", exact(self.audit_prob))
        object.__setattr__(self, "audit_cost", exact(self.audit_cost))
        if not 0 <= self.audit_prob <= 1 or self.audit_cost < 0:
            raise ValueError("need 0 <= audit_prob <= 1 and audit_cost >= 0")

    def reveal_prob(self, action: str, agent: str) -> Fraction:
        """Probability that ``agent``'s outcome is observed after the user takes ``action``."""
        if action == agent:
            return Fraction(1)
        return self.audit_prob if agent == SUBJECT else Fraction(0)

    @property
    def name(self) -> str:
        """``endogenous``, ``exogenous`` or ``audit``."""
        if self.audit_prob == 0:
            return "endogenous"
        if self.audit_prob == 1 and self.audit_cost == 0:
            return "exogenous"
        return "audit"


ENDOGENOUS = Monitoring()
EXOGENOUS = Monitoring(Fraction(1))


@dataclass(frozen=True)
class TransparentAgent:
    """An honest agent of known ability whose report is its own success probability.

    Its draw is coupled to the strategic agent's through the shared latent difficulty, so its report is
    public evidence about the strategic agent's draw.
    """

    name: str
    ability: str

    def __post_init__(self) -> None:
        if self.name in (SUBJECT, SELF):
            raise ValueError(f"{self.name!r} is reserved")


@dataclass(frozen=True)
class Roster:
    """Who the user can delegate to: the strategic agent plus any transparent agents."""

    transparent: tuple[TransparentAgent, ...] = ()

    @property
    def agents(self) -> tuple[str, ...]:
        return (SUBJECT, *(t.name for t in self.transparent))

    @property
    def actions(self) -> tuple[str, ...]:
        """The user's actions: complete the task herself, or delegate to one agent."""
        return (SELF, *self.agents)

    @property
    def is_monopoly(self) -> bool:
        return not self.transparent

    def ability_of(self, name: str) -> str:
        for t in self.transparent:
            if t.name == name:
                return t.ability
        raise KeyError(f"no transparent agent {name!r}")


MONOPOLY = Roster()


@dataclass(frozen=True)
class Game:
    """One fully specified game. See the module docstring for the axes.

    ``δ`` is not part of the game: like the prior, it is a coordinate of the state space the experiments
    sweep, and it is passed to the solvers alongside the belief.
    """

    types: TypeSpace
    payoffs: Payoffs = field(default_factory=Payoffs)
    reports: ReportSpace = field(default_factory=ReportSpace.binary)
    monitoring: Monitoring = ENDOGENOUS
    roster: Roster = MONOPOLY
    horizon: int = 2

    def __post_init__(self) -> None:
        for t in self.roster.transparent:
            if t.ability not in self.types.abilities:
                raise ValueError(f"transparent agent {t.name!r} has unknown ability {t.ability!r}")
        if self.reports.honest_by_draw is not None and set(self.reports.honest_by_draw) != set(self.types.draws):
            raise ValueError("the report space's honest rule must cover every draw")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one period")

    @property
    def rho_star(self) -> Fraction:
        return self.payoffs.rho_star

    @property
    def cost(self) -> Fraction:
        return self.payoffs.cost

    @property
    def cells(self) -> tuple[Cell, ...]:
        """Every ``(η, ability)`` type, honest first."""
        return tuple((eta, a) for eta in (HONEST, STRATEGIC) for a in self.types.abilities)

    def kappa_thresholds(self) -> dict[str, Fraction]:
        """The two myopia thresholds of the binary game: ``κ = 1 − ρ⁺`` and ``κ = 1 − ρ⁻`` (Thm 4.3)."""
        rp, rm = (self.types.rho(d, self.types.abilities[0]) for d in self.types.draws[:2])
        return {"deflation": 1 - rp, "inflation": 1 - rm}

    def delta_star(self) -> Fraction:
        """The myopia threshold in ``δ`` units: ``κ = 1 − ρ⁻``, i.e. ``δ* = 0.459…`` at the paper's values."""
        return delta_of_kappa(self.kappa_thresholds()["inflation"])


def paper_game(**overrides: object) -> Game:
    """The game every experiment in the paper is run at (Table 4).

    Keyword arguments replace axes, e.g. ``paper_game(monitoring=EXOGENOUS)``.
    """
    base = {
        "types": TypeSpace.binary(theta_H="0.8", theta_L="0.2", rho_plus="0.85", rho_minus="0.15"),
        "payoffs": Payoffs(Fraction(1), Fraction(1, 10), Fraction(1, 2)),
        # The signals are the two success probabilities: s ∈ {ρ⁻, ρ⁺}.
        "reports": ReportSpace.binary(low="0.15", high="0.85"),
    }
    base.update(overrides)
    return Game(**base)  # type: ignore[arg-type]
