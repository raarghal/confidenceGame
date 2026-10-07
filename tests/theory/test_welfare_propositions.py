"""The welfare propositions of Appendix P, exact on the joint law at the paper's primitives.

W1 (information destruction is non-negative), W3 (the benchmark ignores the belief), W4 (with correct
population beliefs, knowing the rule never hurts, in either period), the exit cap, W2 (loss factors in
the honest share), and Remark P.4's witness. The paper states them over
``(h, μ) ∈ {0.1, …, 0.9}²``, rules ``{0, ¼, …, 1}²`` and populations equal to the belief or in
``{0, ½, 1}²``; ``--theory-draws`` (any value) runs that full grid, the default a stride of it.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product

import pytest

from strategic_miscalibration.theory.beliefs import Strategy, binary_prior
from strategic_miscalibration.theory.game import paper_game
from strategic_miscalibration.theory.welfare import benchmark, two_periods, user

GAME = paper_game()
OUTSIDE = GAME.payoffs.self_value()
GRID = [F(i, 10) for i in range(1, 10)]
RULES = [(F(a, 4), F(b, 4)) for a in range(5) for b in range(5)]
POPS = [(F(a, 2), F(b, 2)) for a in range(3) for b in range(3)]


def users(h, mu, rule, hs, ms):
    state, pop, r = binary_prior(GAME, h, mu), binary_prior(GAME, hs, ms), Strategy.binary(*rule)
    out = {k: two_periods(GAME, user(GAME, k, state, pop, r), pop, r) for k in ("naive", "soph", "informed")}
    out["bench"] = benchmark(GAME, state, pop)
    return out


@pytest.fixture(scope="module")
def stride(request: pytest.FixtureRequest) -> int:
    return 1 if request.config.getoption("--theory-draws") else 4


def test_propositions(stride: int) -> None:
    for h, mu, rule in product(GRID[::stride], GRID[::stride], RULES):
        for hs, ms in [(h, mu), *POPS]:
            u = users(h, mu, rule, hs, ms)
            b = u["bench"]
            for k in ("naive", "soph", "informed"):  # W1, each period
                assert u[k].user1 <= b.user1 and u[k].user2 <= b.user2
            assert benchmark(GAME, binary_prior(GAME, hs, ms), binary_prior(GAME, hs, ms)) == b  # W3
            if (hs, ms) == (h, mu):
                assert u["soph"].user1 >= u["naive"].user1 and u["soph"].user2 >= u["naive"].user2  # W4
                assert u["soph"].user1 >= OUTSIDE and u["soph"].user2 >= OUTSIDE  # exit cap


def test_w4_strict_count() -> None:
    """The two-period inequality of W4 is strict at 1,439 of the 2,025 (state, rule) pairs.

    The appendix prints 1,464. The count behind it included the state (½, ½) twice for each of the 25
    rules, once as the belief and once among the fixed populations {0, ½, 1}², and 1,439 + 25 = 1,464.
    Recorded in the reproduction report.
    """
    strict = sum(
        u["soph"].user > u["naive"].user
        for h, mu, rule in product(GRID, GRID, RULES)
        for u in [users(h, mu, rule, h, mu)]
    )
    assert strict == 1439


def test_w2_honest_share(stride: int) -> None:
    for h, mu, rule in product(GRID[::stride], GRID[::stride], RULES[::2]):
        for ms in (F(0), F(1, 2), F(1)):
            at_one = users(h, mu, rule, F(1), ms)
            assert at_one["naive"].user == at_one["bench"].user == at_one["informed"].user
            u0 = users(h, mu, rule, F(0), ms)
            l0 = u0["bench"].user - u0["naive"].user
            for hs in (F(1, 4), F(1, 2), F(3, 4)):
                uh = users(h, mu, rule, hs, ms)
                assert uh["bench"].user - uh["naive"].user == (1 - hs) * l0


def test_wrong_population_witness() -> None:
    """Remark P.4: knowing the rule hurts when the population belief is wrong."""
    u = users(F(1, 2), F(1, 2), (F(1), F(1, 2)), F(0), F(1))
    assert (u["naive"].user1, u["naive"].user2) == (F(131, 200), F(61, 100))
    assert (u["soph"].user1, u["soph"].user2) == (F(131, 200), F(11529, 20000))
    assert u["naive"].user - u["soph"].user == F(671, 20000)
