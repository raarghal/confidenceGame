"""Audit of the first-period characterization (Thms 4.2–4.4, Theorem H.9, Theorem I.4).

At random rational primitives, every state on a belief grid and a ladder of κ straddling both myopia
thresholds is solved, and each theorem is checked as the draft scopes it. ``--theory-draws 600`` is the
run the appendix quotes (~264k states); the default is a quick sample.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from strategic_miscalibration.theory.beliefs import Strategy, binary_prior, weights
from strategic_miscalibration.theory.game import SELF, SUBJECT
from strategic_miscalibration.theory.regions import region
from strategic_miscalibration.theory.solvers.binary_t2 import equilibria
from strategic_miscalibration.theory.terminal import psi_star, trust_index

from ..conftest import game_from_prims
from . import bruteforce

HS = [F(1, 20), F(1, 10), F(1, 5), F(3, 10), F(2, 5), F(9, 20), F(11, 20), F(3, 5), F(7, 10), F(4, 5), F(9, 10)]
MUS = [F(1, 10), F(3, 10), F(1, 2), F(7, 10), F(9, 10)]


def od(x):
    return x / (1 - x)


def ge(a, b):  # a >= b, with a = None meaning +∞
    return a is None or a >= b


def indices(game, h, mu, sig):
    law = binary_prior(game, h, mu)
    out = {}
    for s, msg in (("+", "HIGH"), ("-", "LOW")):
        for ev, action, outcomes in (
            ("rej", SELF, {}),
            ("succ", SUBJECT, {SUBJECT: True}),
            ("fail", SUBJECT, {SUBJECT: False}),
        ):
            out[ev + s] = trust_index(
                game, weights(game, law, Strategy.binary(*sig), {SUBJECT: msg}, action, outcomes)
            )
    return out


def edge_closed_form(game, h, mu, kappa):
    """Theorem H.9's conditions (pure terminal play), as stated in the draft."""
    ps, rm = psi_star(game), game.types.rho("hard", "H")
    tb = mu * game.types.p_draw("easy", "H") + (1 - mu) * game.types.p_draw("easy", "L")
    Om = od(tb)
    Psi = Om / (1 - h)
    out = set()
    if kappa > 1 - rm:
        s = Psi / ps
        if Psi < ps:
            idx = indices(game, h, mu, (F(1), s))
            if ge(idx["succ+"], ps) and not ge(idx["rej+"], ps) and ge(idx["rej-"], ps):
                out.add("PS")
        s = 1 + od(h) - Om / ps
        if od(h) < Om / ps < 1 + od(h) and od(h) * Om <= s * ps:
            idx = indices(game, h, mu, (F(0), s))
            if ge(idx["rej+"], ps) and not ge(idx["rej-"], ps) and ge(idx["succ-"], ps) and not ge(idx["fail-"], ps):
                out.add("PI1")
        s = od(h) * Om / ps
        if s < 1 and (1 - h) * Om >= ps * (h + (1 - h) * (1 - s)):
            idx = indices(game, h, mu, (F(0), s))
            if ge(idx["fail+"], ps) and ge(idx["succ-"], ps) and not ge(idx["fail-"], ps):
                out.add("HI1")
    return out


def test_first_period_set(theory_draws: int | None) -> None:
    checked = 0
    for seed in range(theory_draws or 3):
        rng = random.Random(seed)
        tL, tH, rm, rp, rs = bruteforce.rand_prims(rng)
        if rs < F(1, 10):
            continue
        game = game_from_prims(tL, tH, rm, rp, rs)
        kappas = sorted(
            {r * (1 - rm) for r in (F(1, 5), F(1, 2), F(9, 10), F(11, 10), F(3, 2))}
            | {(1 + (1 - rm)) / 2, F(3, 2), F(3)}
        )
        for h in HS:
            for mu in MUS:
                reg = region(game, h, mu)
                for k in kappas:
                    eqs = equilibria(game, h, mu, k / (1 + k))
                    relevant = [p for p in eqs if p.payoff_relevant]
                    pure = [p for p in relevant if p.terminal_mix is None]
                    where = (game.types.draw_probs, h, mu, k)
                    # Thm 4.2: truthful play is never a payoff-relevant equilibrium.
                    assert not any(p.sigma == (1, 0) for p in relevant), where
                    # Thm 4.3: with pure terminal play and the standard user, the set is exactly the prediction.
                    if reg != "out":
                        std = {p.sigma for p in pure if p.user == (1, 0)}
                        predicted = (reg == "R") or (reg == "F" and k >= 1 - rm) or (reg == "D" and k > 1)
                        assert std == ({(F(1), F(1))} if predicted else set()), where
                        # Above the threshold on τF the inflation equilibrium SD2 is the only one.
                        if reg == "F" and k > 1 - rm:
                            assert all(p.user == (1, 0) and p.sigma == (1, 1) for p in relevant), where
                    # Thm 4.4 (pure terminal): under-reporting needs σ = (0, x), h < 1/2, κ > 1 − ρ⁻, one mixed user.
                    for p in pure:
                        if p.sigma_plus < 1:
                            assert p.sigma_plus == 0 and 0 < p.sigma_minus < 1, where
                            assert h < F(1, 2) and k > 1 - rm, where
                            assert (0 < p.d_plus < 1) + (0 < p.d_minus < 1) == 1, where
                    # Theorem H.9: the closed form and the solver agree.
                    solver_edge = {p.family[5:] for p in pure if p.family.startswith("edge:")}
                    assert edge_closed_form(game, h, mu, k) == solver_edge, where
                    checked += 1
    assert checked > 0


def test_watershed_by_solver(theory_draws: int | None) -> None:
    """Thm 4.4(ii): at h > 1/2 every payoff-relevant equilibrium has σ⁺ = 1."""
    rng = random.Random(99)
    for _ in range(theory_draws or 10):
        tL, tH, rm, rp, rs = bruteforce.rand_prims(rng)
        if rs < F(1, 10):
            continue
        game = game_from_prims(tL, tH, rm, rp, rs)
        for h in (F(11, 20), F(3, 5), F(4, 5), F(19, 20)):
            for mu in (F(1, 10), F(1, 2), F(9, 10)):
                for k in (F(1, 10), F(1, 2), F(1), F(3)):
                    for p in equilibria(game, h, mu, k / (1 + k)):
                        assert not p.payoff_relevant or p.sigma_plus == 1


def test_watershed_by_brute_force(theory_draws: int | None) -> None:
    """The same claim, independently: pure users × an agent grid, through the enumerator only."""
    import itertools

    rng = random.Random(99)
    grid = [F(k, 4) for k in range(5)]
    for _ in range(theory_draws or 6):
        g = bruteforce.Game(*bruteforce.rand_prims(rng))
        for h in (F(11, 20), F(4, 5)):
            for mu in (F(1, 10), F(9, 10)):
                for sp, sm in itertools.product(grid, grid):
                    if sp == 1:
                        continue
                    for dp, dm in itertools.product((0, 1), (0, 1)):
                        for k in (F(1, 10), F(1), F(3)):
                            r = g.is_eq(h, mu, (sp, sm), {"+": dp, "-": dm}, k)
                            if r is not None:
                                assert not (r[0] and (r[1]["+"] != 0 or r[1]["-"] != 0))
