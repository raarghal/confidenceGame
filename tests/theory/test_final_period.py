"""Thms 4.1 and 4.2: the final-period identities the proofs use, checked exactly at random joint laws.

Each claim is checked against the Bayes engine (and, for the trust test, the independent enumerator).
Paper references are to the compiled paper: Eq. (17) and the mean identity are in the
beliefs appendix, the trust index and blind-trust maximin in the proof of Thm 4.1, the ordering in the
proof of Thm 4.2.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from strategic_miscalibration.theory.beliefs import (
    Strategy,
    binary_prior,
    expected_success,
    honesty,
    posterior,
    report_probability,
)
from strategic_miscalibration.theory.game import SELF, SUBJECT
from strategic_miscalibration.theory.terminal import is_trusted, trust_index

from . import bruteforce
from ._laws import frac, od, random_game, random_law

MSG = {"+": "HIGH", "-": "LOW"}


def theta_tilde(game, law, rule, s):
    """P(easy | report s), recovered from the engine's E[success | s] = ρ⁻ + (ρ⁺ − ρ⁻) P(easy | s)."""
    rp, rm = game.types.rho("easy", "H"), game.types.rho("hard", "H")
    rt = expected_success(game, law, rule, {SUBJECT: MSG[s]})
    return None if rt is None else (rt - rm) / (rp - rm)


def moments(game, law):
    th = {a: game.types.p_draw("easy", a) for a in ("H", "L")}
    G = sum(w * th[a] for (_, a), w in law.items())
    B = sum(w * (1 - th[a]) for (eta, a), w in law.items() if eta == 0)
    easy_honest = sum(w * th[a] for (eta, a), w in law.items() if eta == 1)
    hard_honest = sum(w * (1 - th[a]) for (eta, a), w in law.items() if eta == 1)
    return G, B, easy_honest / G, hard_honest / (1 - G)


def test_odds_mean_identity_and_trust_index() -> None:
    rng = random.Random(7)
    for _ in range(150):
        game, law = random_game(rng), random_law(rng)
        G, B, h_easy, h_hard = moments(game, law)
        assert trust_index(game, law) == G / B == od(G) / (1 - h_hard)
        for sp, sm in ((F(1), F(1)), (F(1), F(0)), (F(0), F(1)), (F(0), F(0)), (frac(rng), frac(rng))):
            rule = Strategy.binary(sp, sm)
            tp, tm = theta_tilde(game, law, rule, "+"), theta_tilde(game, law, rule, "-")
            if sm > 0:  # Eq. (17), high report
                assert od(tp) == od(G) * (h_easy + (1 - h_easy) * sp) / ((1 - h_hard) * sm)
            elif tp is not None:
                assert tp == 1
            if tm is not None:  # Eq. (17), low report
                assert od(tm) == od(G) * (1 - h_easy) * (1 - sp) / (h_hard + (1 - h_hard) * (1 - sm))
            p_hi = report_probability(game, law, rule, {SUBJECT: "HIGH"})
            mean = p_hi * (tp or 0) + (1 - p_hi) * (tm or 0)
            assert mean == G  # P(high) θ̃(+) + P(low) θ̃(−) = P(easy)
        assert od(theta_tilde(game, law, Strategy.binary(1, 1), "+")) == G / B  # Ψ = od θ̃(+) under (1, 1)


def test_trust_test_matches_independent_enumerator() -> None:
    rng = random.Random(5)
    for _ in range(400):
        game, law = random_game(rng), random_law(rng)
        tL, tH = game.types.p_draw("easy", "L"), game.types.p_draw("easy", "H")
        rp, rm = game.types.rho("easy", "H"), game.types.rho("hard", "H")
        bf = bruteforce.Game(tL, tH, rm, rp, game.rho_star)
        assert is_trusted(game, law) == (bf.V2(law) == bf.c)


def test_blind_trust_maximin_and_nesting() -> None:
    """Max over rules of min over reports of od θ̃ is od(G)·min(1, (1 − h_easy)/h_hard); τ_B ⊂ int τ_C."""
    rng = random.Random(11)
    grid = [F(i, 10) for i in range(11)]
    for _ in range(40):
        game, law = random_game(rng), random_law(rng)
        G, B, h_easy, h_hard = moments(game, law)

        def min_odds(sp, sm, game=game, law=law):
            rule = Strategy.binary(sp, sm)
            vals = [theta_tilde(game, law, rule, s) for s in "+-"]
            return min(od(v) for v in vals if v is not None and v < 1)

        claim = od(G) * min(F(1), (1 - h_easy) / h_hard)
        if h_easy + h_hard <= 1:  # attained by the jamming rule P(high | easy) = P(high | hard)
            assert min_odds(F(0), h_easy / (1 - h_hard)) == claim == od(G)
        else:
            assert min_odds(F(0), F(1)) == claim
        assert max(min_odds(a, b) for a in grid for b in grid) <= claim
        assert od(G) < G / B  # τ_B lies in the interior of τ_C when h > 0


def test_truthful_reports_keep_honesty_and_order_trust() -> None:
    """Thm 4.2, first period: truthful play from a product prior leaves h unchanged and Ψ(π⁻) < Ψ(π⁺)."""
    rng = random.Random(922)
    truthful = Strategy.binary(1, 0)
    for _ in range(200):
        game = random_game(rng)
        law = binary_prior(game, frac(rng), frac(rng))
        after = {s: posterior(game, law, truthful, {SUBJECT: MSG[s]}) for s in "+-"}
        assert honesty(after["+"]) == honesty(after["-"]) == honesty(law)
        assert trust_index(game, after["-"]) < trust_index(game, after["+"])


def test_truthful_order_after_any_first_period_event() -> None:
    """Truthful reports order the trust index, at t = 2 of T = 2 (proof of Thm 4.2(iv)).

    From a product prior, after *any* first-period event under *any* rule, a truthful low report in the
    next period is trusted strictly less than a truthful high report (or the strategic mass is gone).
    """
    rng = random.Random(100)
    truthful = Strategy.binary(1, 0)
    events = ((SELF, {}), (SUBJECT, {SUBJECT: True}), (SUBJECT, {SUBJECT: False}))
    for _ in range(300):
        game = random_game(rng)
        law = binary_prior(game, frac(rng), frac(rng))
        rule = Strategy.binary(*(rng.choice([F(0), F(1), frac(rng)]) for _ in range(2)))
        for msg in ("HIGH", "LOW"):
            for action, outcomes in events:
                mid = posterior(game, law, rule, {SUBJECT: msg}, action, outcomes)
                if mid is None:
                    continue
                lo = trust_index(game, posterior(game, mid, truthful, {SUBJECT: "LOW"}))
                hi = trust_index(game, posterior(game, mid, truthful, {SUBJECT: "HIGH"}))
                if lo is None and hi is None:
                    continue  # h = 1 after the event: both +∞
                assert hi is None or (lo is not None and lo < hi)


def test_honesty_selection_example() -> None:
    """Example F.2: at (87/100, 1/1000) truthful play survives only a non-agent-optimal selection.

    After a truthful high report the terminal user can strictly reject under rule (0, 1), while the low
    report stays trusted, so the theorem needs the agent-optimal terminal selection.
    """
    from ..conftest import game_from_prims

    game = game_from_prims(F(1, 5), F(4, 5), F(3, 20), F(17, 20), F(3, 5))
    law = binary_prior(game, F(87, 100), F(1, 1000))
    truthful = Strategy.binary(1, 0)
    pi_hi = posterior(game, law, truthful, {SUBJECT: "HIGH"})
    pi_lo = posterior(game, law, truthful, {SUBJECT: "LOW"})
    assert is_trusted(game, pi_lo)
    inverted = Strategy.binary(0, 1)
    for msg in ("HIGH", "LOW"):
        assert expected_success(game, pi_hi, inverted, {SUBJECT: msg}) < game.rho_star
