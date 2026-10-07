"""The solver against the independent enumerator, at random primitives (``joint_solver_crosscheck``).

Corner and edge profiles (terminal values in {0, c}) must be certified as equilibria by brute force, with the
same payoff-relevance verdict. Boundary profiles must put the pinned belief exactly on the trust boundary
by brute force, and the enumerator's gains, with that belief's value set to the solver's ``α·c``, must equal
the solver's.
"""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction as F

from strategic_miscalibration.theory.beliefs import Strategy, binary_prior, normalize, weights
from strategic_miscalibration.theory.game import SELF, SUBJECT, kappa
from strategic_miscalibration.theory.solvers.binary_t2 import equilibria

from ..conftest import game_from_prims
from . import bruteforce

KEYS = ("rej+", "succ+", "fail+", "rej-", "succ-", "fail-")


def _belief(game, law, sig, key):
    action, outcomes = (SELF, {}) if key.startswith("rej") else (SUBJECT, {SUBJECT: key.startswith("succ")})
    return normalize(
        weights(game, law, Strategy.binary(*sig), {SUBJECT: "HIGH" if key[-1] == "+" else "LOW"}, action, outcomes)
    )


def test_solver_agrees_with_enumerator(theory_draws: int | None) -> None:
    rng = random.Random(77)
    found: Counter = Counter()
    for _ in range(theory_draws or 8):
        tL, tH, rm, rp, rs = bruteforce.rand_prims(rng)
        if rs < F(1, 10):
            continue
        bf, game = bruteforce.Game(tL, tH, rm, rp, rs), game_from_prims(tL, tH, rm, rp, rs)
        for h in (F(1, 10), F(3, 10), F(1, 2), F(7, 10)):
            for mu in (F(1, 5), F(1, 2), F(4, 5)):
                for k in (F(1, 6), F(1, 2), F(1), F(5, 2)):
                    delta = k / (1 + k)
                    assert kappa(delta) == k
                    for p in equilibria(game, h, mu, delta):
                        found[(p.family.split(":")[0], p.payoff_relevant)] += 1
                        sig, d = p.sigma, {"+": p.d_plus, "-": p.d_minus}
                        if p.terminal_mix is None:
                            r = bf.is_eq(h, mu, sig, d, k)
                            assert r and r[0]
                            assert (r[1]["+"] != 0 or r[1]["-"] != 0) == p.payoff_relevant
                            continue
                        key = p.terminal_mix[0].split("/")[0]
                        post = bf.successor(
                            bf.prior(h, mu), sig, key[-1] == "+", {"rej": None, "succ": True, "fail": False}[key[:-1]]
                        )
                        assert bf.posterior_success(post, (F(1), F(1)), True) == bf.rs
                        V = bf.values(h, mu, sig)
                        law = binary_prior(game, h, mu)
                        pinned = _belief(game, law, sig, key)
                        for kk in KEYS:
                            if _belief(game, law, sig, kk) == pinned:
                                V[kk] = p.values[kk]
                        D = bf.gains(V, d, delta)
                        assert (D["+"], D["-"]) == (p.delta_plus, p.delta_minus)
    # Every family must actually be exercised, or the test is vacuous.
    assert {fam for fam, _ in found} == {"corner", "edge", "boundary"}
