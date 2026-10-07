"""First-period equilibria of the paper's two-period game, in exact arithmetic.

Covers: a monopoly, binary reports, the paper's ungraded binary type space, endogenous monitoring, two
periods. Every belief is the joint law on the four type cells, and every terminal value comes from the
final-period trust test of :mod:`..terminal`, so nothing here assumes the ``(h, μ)`` marginals are a
sufficient state (they are not after a non-honest report).

When an agent's mixing puts a terminal belief exactly on the trust boundary, the terminal user may
delegate with any probability, so that belief's value is free in ``[0, c]``. This is how the boundary
equilibria of Thm 4.4 exist below the myopia threshold.

Every candidate equation is affine in its single unknown, so roots, best-response signs and payoff
relevance are decided exactly. Families enumerated (complete at generic parameters, except users who mix
after both reports):

``corner``     user and agent both pure (16 profiles); terminal values in ``{0, c}``.
``edge``       the user mixes after exactly one report and the agent mixes on one draw. The agent's mix is
               pinned by the user's indifference and the user's mix by the agent's. Recovers PS, PI1, HI1
               and HS of Theorem H.9 without their closed forms.
``boundary``   the user is pure and the agent mixes on one draw; the mix is pinned by a terminal belief on
               the trust boundary, and that belief's value by the agent's indifference. Recovers
               Proposition H.5's ``(1, σ̄)`` and the second boundary family.

Not enumerated: users who mix after both reports (on the jamming locus; payoff-irrelevant under
Assumption P); agents who mix on both draws (always payoff-irrelevant); and non-generic double
coincidences, which are recorded in ``Profile.flags`` rather than dropped silently.

``tests/theory`` cross-checks it against recorded reference equilibria and an independent brute-force
enumerator.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from fractions import Fraction
from functools import partial

from ...core.numbers import Number, exact
from ..beliefs import Law, Strategy, binary_prior, normalize, success_moments, weights
from ..game import SELF, SUBJECT, Game
from ..terminal import final_success_gap
from .base import SOLVERS, Profile

__all__ = ["BinaryTwoPeriod", "agent_optimal", "equilibria", "payoff_relevant", "sigma_bar_closed_form"]

REPORTS = ("+", "-")
EVENTS = ("rej", "succ", "fail")
MESSAGE = {"+": "HIGH", "-": "LOW"}
KEYS = tuple(e + s for s in REPORTS for e in EVENTS)

# An agent mixing on one draw: (which coordinate mixes, the other coordinate's value).
AGENT_SHAPES = (
    ("sigma_minus", Fraction(1)),
    ("sigma_minus", Fraction(0)),
    ("sigma_plus", Fraction(1)),
    ("sigma_plus", Fraction(0)),
)
EDGE_NAMES = {("+", 0): "PS", ("+", 1): "HI1", ("-", 0): "PI1", ("-", 1): "HS"}

Sig = tuple[Fraction, Fraction]


def _strategy(sig: Sig) -> Strategy:
    return Strategy.binary(sig[0], sig[1])


def _sig_of(shape: tuple[str, Fraction], x: Fraction) -> Sig:
    coord, fixed = shape
    return (fixed, x) if coord == "sigma_minus" else (x, fixed)


# Per-state memo: the enumeration revisits the same rule many times (16 corner profiles share 4 rules).
# Keyed on the law's identity, which is fixed for the duration of one `equilibria` call.
_MEMO: dict[tuple, object] = {}


def _memo(kind: str, law: Law, sig: Sig, key: str, compute: Callable[[], object]) -> object:
    k = (kind, id(law), sig, key)
    if k not in _MEMO:
        _MEMO[k] = compute()
    return _MEMO[k]


def _event_weights(game: Game, law: Law, sig: Sig, key: str) -> dict:
    """Weights of the terminal belief after report ``key[-1]`` and event ``key[:-1]``."""
    event, report = key[:-1], key[-1]
    action, outcomes = (SELF, {}) if event == "rej" else (SUBJECT, {SUBJECT: event == "succ"})
    return _memo(  # type: ignore[return-value]
        "w", law, sig, key, lambda: weights(game, law, _strategy(sig), {SUBJECT: MESSAGE[report]}, action, outcomes)
    )


def _gaps(game: Game, law: Law, sig: Sig) -> dict[str, Fraction | None]:
    """Final-period trust gap at each of the six terminal beliefs (``None`` = no bluffable mass: trusted)."""
    return _memo(  # type: ignore[return-value]
        "g", law, sig, "", lambda: {k: final_success_gap(game, _event_weights(game, law, sig, k)) for k in KEYS}
    )


def _value(game: Game, gap: Fraction | None) -> Fraction:
    return game.cost if gap is None or gap >= 0 else Fraction(0)


def _rho_tilde(game: Game, law: Law, sig: Sig, report: str) -> Fraction | None:
    num, den = success_moments(game, law, _strategy(sig), {SUBJECT: MESSAGE[report]})
    return None if den == 0 else num / den


def _rt_gap(game: Game, law: Law, sig: Sig, report: str) -> Fraction:
    """``P(report) · (ρ̃ − ρ*)``: affine in the agent's mix."""
    num, den = success_moments(game, law, _strategy(sig), {SUBJECT: MESSAGE[report]})
    return num - game.rho_star * den


def _draw_rho(game: Game) -> dict[str, Fraction]:
    a = game.types.abilities[0]
    return {"+": game.types.rho("easy", a), "-": game.types.rho("hard", a)}


def delta_gain(game: Game, delta: Fraction, V: dict[str, Fraction], d: dict[str, Fraction], rho: Fraction) -> Fraction:
    """``Δ(ρ)``: discounted payoff of reporting high minus low at true success probability ``ρ``.

    ``δ`` weights this period's fee and ``1 − δ`` the continuation, as in the prompts.
    """

    def W(s: str) -> Fraction:
        delegated = rho * V["succ" + s] + (1 - rho) * V["fail" + s]
        return d[s] * (delta * game.cost + (1 - delta) * delegated) + (1 - d[s]) * (1 - delta) * V["rej" + s]

    return W("+") - W("-")


def _user_ok(game: Game, rt: Fraction | None, dd: Fraction) -> bool:
    if rt is None or rt == game.rho_star:
        return True
    return dd == (1 if rt > game.rho_star else 0)


def _agent_ok(sig_coord: Fraction, gain: Fraction) -> bool:
    if gain == 0:
        return True
    return sig_coord == (1 if gain > 0 else 0)


def _ex_ante_agent_value(
    game: Game, law: Law, delta: Fraction, V: dict[str, Fraction], d: dict[str, Fraction], sig: Sig
) -> Fraction:
    """The strategic agent's value before its draw, weighted by the prior over its ability."""
    rho = _draw_rho(game)
    total = Fraction(0)
    for a in game.types.abilities:
        pa = sum((v for (_, ab), v in law.items() if ab == a), Fraction(0))
        for draw, s_hi in (("+", sig[0]), ("-", sig[1])):
            pd = game.types.p_draw("easy" if draw == "+" else "hard", a)
            for s, ps in (("+", s_hi), ("-", 1 - s_hi)):
                delegated = rho[draw] * V["succ" + s] + (1 - rho[draw]) * V["fail" + s]
                w = d[s] * (delta * game.cost + (1 - delta) * delegated) + (1 - d[s]) * (1 - delta) * V["rej" + s]
                total += pa * pd * ps * w
    return total


def _finish(
    game: Game,
    law: Law,
    delta: Fraction,
    d: dict[str, Fraction],
    sig: Sig,
    V: dict[str, Fraction],
    family: str,
    terminal_mix: tuple[str, Fraction] | None = None,
    flags: list[str] | None = None,
) -> Profile | None:
    """Keep the profile iff both players best-respond."""
    if not all(_user_ok(game, _rho_tilde(game, law, sig, s), d[s]) for s in REPORTS):
        return None
    rho = _draw_rho(game)
    gp, gm = delta_gain(game, delta, V, d, rho["+"]), delta_gain(game, delta, V, d, rho["-"])
    if not (_agent_ok(sig[0], gp) and _agent_ok(sig[1], gm)):
        return None
    value = _ex_ante_agent_value(game, law, delta, V, d, sig)
    relevant = gp != 0 or gm != 0
    return Profile(d["+"], d["-"], sig[0], sig[1], family, relevant, gp, gm, value, dict(V), terminal_mix, flags or [])


def _linear_root(f: Callable[[Fraction], Fraction]) -> Fraction | None:
    """Root of a function affine on ``[0, 1]``; ``None`` if it is flat."""
    f0, f1 = f(Fraction(0)), f(Fraction(1))
    return None if f0 == f1 else f0 / (f0 - f1)


def _user_gap_at(game: Game, law: Law, shape: tuple[str, Fraction], report: str, t: Fraction) -> Fraction:
    """The user's indifference condition after ``report`` when the agent's mix is ``t``."""
    return _rt_gap(game, law, _sig_of(shape, t), report)


def _pin_gap_at(game: Game, law: Law, shape: tuple[str, Fraction], key: str, t: Fraction) -> Fraction:
    """The terminal trust gap at belief ``key`` when the agent's mix is ``t`` (zero mass reads as 0)."""
    gap = final_success_gap(game, _event_weights(game, law, _sig_of(shape, t), key))
    return Fraction(0) if gap is None else gap


def _edge_gain(
    game: Game, delta: Fraction, V: dict[str, Fraction], mixed: str, d_other: Fraction, rho: Fraction, m: Fraction
) -> Fraction:
    """The mixing draw's gain when the user delegates ``mixed`` with probability ``m``."""
    return delta_gain(game, delta, V, {mixed: m, _other(mixed): d_other}, rho)


def _boundary_gain(
    game: Game,
    delta: Fraction,
    base: dict[str, Fraction],
    free: list[str],
    d: dict[str, Fraction],
    rho: Fraction,
    v: Fraction,
) -> Fraction:
    """The mixing draw's gain when the boundary beliefs are worth ``v·c``."""
    V = dict(base)
    V.update(dict.fromkeys(free, v * game.cost))
    return delta_gain(game, delta, V, d, rho)


def _other(report: str) -> str:
    return "-" if report == "+" else "+"


def corner_profiles(game: Game, law: Law, delta: Fraction) -> list[Profile]:
    out = []
    for dp, dm, sp, sm in itertools.product((0, 1), repeat=4):
        sig = (Fraction(sp), Fraction(sm))
        d = {"+": Fraction(dp), "-": Fraction(dm)}
        gaps = _gaps(game, law, sig)
        flags = ["on-boundary:" + k for k, g in gaps.items() if g == 0]
        V = {k: _value(game, g) for k, g in gaps.items()}
        prof = _finish(game, law, delta, d, sig, V, "corner", flags=flags)
        if prof is not None:
            out.append(prof)
    return out


def edge_profiles(game: Game, law: Law, delta: Fraction) -> list[Profile]:
    """The user mixes after exactly one report; the agent mixes on one draw, pinned by the user's indifference."""
    out = []
    rho = _draw_rho(game)
    for mixed in REPORTS:
        other = _other(mixed)
        for d_other in (Fraction(0), Fraction(1)):
            for shape in AGENT_SHAPES:
                x = _linear_root(partial(_user_gap_at, game, law, shape, mixed))
                if x is None or not 0 < x < 1:
                    continue
                sig = _sig_of(shape, x)
                if _rho_tilde(game, law, sig, mixed) != game.rho_star:
                    continue
                gaps = _gaps(game, law, sig)
                flags = ["on-boundary:" + k for k, g in gaps.items() if g == 0]
                V = {k: _value(game, g) for k, g in gaps.items()}
                rho_int = rho["-"] if shape[0] == "sigma_minus" else rho["+"]

                gain = partial(_edge_gain, game, delta, V, mixed, d_other, rho_int)
                m = _linear_root(gain)
                if m is None:
                    if gain(Fraction(0)) == 0:
                        flags.append("agent-indifferent-for-every-user-mix")
                    continue
                if not 0 < m < 1:
                    continue
                d = {mixed: m, other: d_other}
                name = EDGE_NAMES[(mixed, int(d_other))]
                prof = _finish(game, law, delta, d, sig, V, f"edge:{name}", flags=flags)
                if prof is not None:
                    out.append(prof)
    return out


def boundary_profiles(game: Game, law: Law, delta: Fraction) -> list[Profile]:
    """Pure user; the agent mixes on one draw, pinned by a terminal belief on the trust boundary."""
    out = []
    rho = _draw_rho(game)
    for dp, dm in itertools.product((0, 1), repeat=2):
        d = {"+": Fraction(dp), "-": Fraction(dm)}
        for shape in AGENT_SHAPES:
            for key in KEYS:
                x = _linear_root(partial(_pin_gap_at, game, law, shape, key))
                if x is None or not 0 < x < 1:
                    continue
                sig = _sig_of(shape, x)
                gaps = _gaps(game, law, sig)
                if gaps[key] != 0:
                    continue
                groups = {k: normalize(_event_weights(game, law, sig, k)) for k in KEYS}
                free = [k for k in KEYS if groups[k] == groups[key]]
                flags = ["second-boundary:" + k for k, g in gaps.items() if k not in free and g == 0]
                base = {k: _value(game, g) for k, g in gaps.items()}
                rho_int = rho["-"] if shape[0] == "sigma_minus" else rho["+"]

                a = _linear_root(partial(_boundary_gain, game, delta, base, free, d, rho_int))
                if a is None or not 0 <= a <= 1:
                    continue
                V = dict(base)
                V.update(dict.fromkeys(free, a * game.cost))
                label = "/".join(sorted(free))
                prof = _finish(game, law, delta, d, sig, V, f"boundary:{label}", terminal_mix=(label, a), flags=flags)
                if prof is not None:
                    out.append(prof)
    return out


class BinaryTwoPeriod:
    """The paper's solver; see the module docstring."""

    def unsupported(self, game: Game) -> str | None:
        t = game.types
        if not game.roster.is_monopoly:
            return "needs a monopoly (no transparent agents)"
        if game.reports.messages != ("LOW", "HIGH") or game.reports.honest_by_draw is None:
            return "needs the binary LOW/HIGH report space"
        if t.draws != ("easy", "hard") or t.success_depends_on_ability:
            return "needs the ungraded binary type space"
        if game.monitoring.audit_prob != 0:
            return "needs endogenous monitoring"
        if game.horizon != 2:
            return "needs a two-period horizon"
        return None

    def equilibria(self, game: Game, h: Number, mu: Number, delta: Number) -> list[Profile]:
        why = self.unsupported(game)
        if why:
            raise NotImplementedError(why)
        law, dl = binary_prior(game, h, mu), exact(delta)
        try:
            found = corner_profiles(game, law, dl) + edge_profiles(game, law, dl) + boundary_profiles(game, law, dl)
        finally:
            _MEMO.clear()
        seen, out = set(), []
        for prof in found:
            key = (prof.d_plus, prof.d_minus, prof.sigma_plus, prof.sigma_minus, prof.terminal_mix)
            if key not in seen:
                seen.add(key)
                out.append(prof)
        return out


SOLVERS.add("binary_t2", BinaryTwoPeriod())


def equilibria(game: Game, h: Number, mu: Number, delta: Number) -> list[Profile]:
    """All enumerated first-period equilibria at ``(h, μ, δ)``, de-duplicated."""
    return BinaryTwoPeriod().equilibria(game, h, mu, delta)


def payoff_relevant(game: Game, h: Number, mu: Number, delta: Number) -> list[Profile]:
    """The equilibria in which the report changes the agent's payoff (Definition C.6)."""
    return [p for p in equilibria(game, h, mu, delta) if p.payoff_relevant]


def agent_optimal(profiles: list[Profile]) -> list[Profile]:
    """The agent-preferred members (Definition C.4); ties kept."""
    if not profiles:
        return []
    best = max(p.agent_value for p in profiles)
    return [p for p in profiles if p.agent_value == best]


def sigma_bar_closed_form(game: Game, h: Number, mu: Number) -> Fraction:
    """Proposition H.5: the bluffing rate of the boundary family ``(1, σ̄)``. Used by tests, not the enumeration."""
    from ..terminal import psi_star

    hh, m = exact(h), exact(mu)
    tH, tL = (game.types.p_draw("easy", a) for a in game.types.abilities[:2])

    def E(f: Callable[[Fraction], Fraction]) -> Fraction:
        return m * f(tH) + (1 - m) * f(tL)

    e2, mix, q = E(lambda t: t * t), E(lambda t: t * (1 - t)), E(lambda t: (1 - t) ** 2)
    rho = _draw_rho(game)
    lp, lm = 1 - rho["+"], 1 - rho["-"]
    ps = psi_star(game)
    return (lp / lm) * (e2 / (1 - hh) - ps * mix) / (ps * q - mix)
