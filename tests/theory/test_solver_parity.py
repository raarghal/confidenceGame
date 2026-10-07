"""The solver reproduces recorded reference equilibria, profile for profile.

The reference values were computed once by the solver this one was rewritten from, and saved in
``tests/fixtures/theory/equilibria_joint.json``: the paper's 100-state grid at four discount factors,
and random rational primitives.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from strategic_miscalibration.theory.solvers.binary_t2 import equilibria

from ..conftest import FIXTURES, game_from_prims

CASES = json.loads((FIXTURES / "theory" / "equilibria_joint.json").read_text())


def _key(p: dict) -> tuple:
    return (*p["d"], *p["sigma"], p["family"], json.dumps(p["terminal_mix"]))


def _as_dict(p) -> dict:
    return {
        "d": [str(p.d_plus), str(p.d_minus)],
        "sigma": [str(p.sigma_plus), str(p.sigma_minus)],
        "family": p.family,
        "payoff_relevant": p.payoff_relevant,
        "delta_gain": [str(p.delta_plus), str(p.delta_minus)],
        "agent_value": str(p.agent_value),
        "values": {k: str(v) for k, v in p.values.items()},
        "terminal_mix": None if p.terminal_mix is None else [p.terminal_mix[0], str(p.terminal_mix[1])],
    }


def _usable(case: dict) -> bool:
    # Payoffs needs 0 < c < e <= r with r = 1, i.e. rho* >= c.
    return Fraction(case["prims"]["rho_star"]) >= Fraction(case["prims"]["c"])


@pytest.mark.parametrize("case", [c for c in CASES if _usable(c)], ids=lambda c: f"{c['h']},{c['mu']},{c['delta']}")
def test_profiles_match_reference(case: dict) -> None:
    P = {k: Fraction(v) for k, v in case["prims"].items()}
    game = game_from_prims(P["theta_L"], P["theta_H"], P["rho_minus"], P["rho_plus"], P["rho_star"], P["c"])
    got = {_key(d): d for d in map(_as_dict, equilibria(game, case["h"], case["mu"], case["delta"]))}
    want = {_key(d): d for d in case["profiles"]}
    assert got.keys() == want.keys()
    for k in want:
        assert got[k] == want[k]


def test_fixture_covers_the_paper_grid() -> None:
    assert sum(_usable(c) for c in CASES) >= 300
