"""The first-period closed forms of the beliefs appendix ("First-period closed forms"), exact at random points.

Eq. (22), Eq. (23), Eq. (24), Eq. (25) and the moments behind Eq. (35),
each checked against the Bayes engine's posterior and trust index.
"""

from __future__ import annotations

import random

from strategic_miscalibration.theory.beliefs import Strategy, binary_prior, expected_success, posterior
from strategic_miscalibration.theory.game import SELF, SUBJECT
from strategic_miscalibration.theory.terminal import trust_index

from ._laws import frac, od, random_game

EVENT = {None: (SELF, {}), 1: (SUBJECT, {SUBJECT: True}), 0: (SUBJECT, {SUBJECT: False})}


def test_closed_forms() -> None:
    rng = random.Random(1)
    for _ in range(250):
        game = random_game(rng)
        h, mu, x = frac(rng), frac(rng), frac(rng)
        tH, tL = game.types.p_draw("easy", "H"), game.types.p_draw("easy", "L")
        rp, rm = game.types.rho("easy", "H"), game.types.rho("hard", "H")
        law = binary_prior(game, h, mu)
        tb = mu * tH + (1 - mu) * tL
        Om = od(tb)

        def E(f, mu=mu, tH=tH, tL=tL):
            return mu * f(tH) + (1 - mu) * f(tL)

        e2, m, q = E(lambda t: t * t), E(lambda t: t * (1 - t)), E(lambda t: (1 - t) ** 2)
        lp, lm = 1 - rp, 1 - rm

        def Psi(sp, sm, s, o, game=game, law=law):
            action, outcomes = EVENT[o]
            post = posterior(game, law, Strategy.binary(sp, sm), {SUBJECT: s}, action, outcomes)
            return trust_index(game, post)

        def odds_easy(sp, sm, s, game=game, law=law, rp=rp, rm=rm):
            rt = expected_success(game, law, Strategy.binary(sp, sm), {SUBJECT: s})
            return od((rt - rm) / (rp - rm))

        for sp, sm in ((1, x), (0, x), (x, 0), (frac(rng), frac(rng))):  # Eq. (22)
            if sm > 0:
                assert odds_easy(sp, sm, "HIGH") == Om * (h + (1 - h) * sp) / ((1 - h) * sm)
            assert odds_easy(sp, sm, "LOW") == Om * (1 - h) * (1 - sp) / (h + (1 - h) * (1 - sm))

        # rule (1, x): Eq. (23) and Eq. (24)
        assert Psi(1, x, "HIGH", 0) == (lp * e2 + (1 - h) * lm * m * x) / ((1 - h) * (lp * m + lm * q * x))
        assert Psi(1, x, "HIGH", 1) == (rp * e2 + (1 - h) * rm * m * x) / ((1 - h) * (rp * m + rm * q * x))
        assert Psi(1, x, "HIGH", None) == (e2 + (1 - h) * m * x) / ((1 - h) * (m + q * x))
        th_hard, hL = m / (1 - tb), h / (h + (1 - h) * (1 - x))
        assert Psi(1, x, "LOW", None) == od(th_hard) / (1 - hL)
        for o in (0, 1):  # a low report under σ⁺ = 1 is outcome-inert
            assert Psi(1, x, "LOW", o) == Psi(1, x, "LOW", None)

        # rule (0, x): Eq. (25)
        assert Psi(0, x, "HIGH", None) == (od(h) * e2 + m * x) / (q * x)
        for o, lik_p, lik_m in ((1, rp, rm), (0, lp, lm)):
            assert Psi(0, x, "HIGH", o) == (od(h) * lik_p * e2 + x * lik_m * m) / (x * lik_m * q)
            assert Psi(0, x, "LOW", o) == (od(h) * lik_m * m + lik_p * e2 + (1 - x) * lik_m * m) / (
                lik_p * m + (1 - x) * lik_m * q
            )
        assert Psi(0, x, "LOW", None) == (od(h) * m + e2 + (1 - x) * m) / (m + (1 - x) * q)

        # the moments behind Eq. (35) are the x = 1 case
        a = E(lambda t, rp=rp, rm=rm: t * (1 - (t * rp + (1 - t) * rm)))
        b = E(lambda t, rp=rp, rm=rm: (1 - t) * (1 - (t * rp + (1 - t) * rm)))
        assert a == lp * e2 + lm * m
        assert b == lp * m + lm * q
