"""The boundary families SD3, SD3′, SD6 and BI1–BI3 (Proposition I.1, Example I.2).

Every instance is checked twice: by the independent enumerator (the pinned belief is exactly on the trust
boundary, both users' gates pass, the agent's gains have the stated signs, every other inequality is strict
so the instance is generic, the pinning function has nonzero slope) and by the solver, which must enumerate
an equilibrium with exactly this rule, user and terminal mix.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from strategic_miscalibration.theory.solvers.binary_t2 import equilibria

from ..conftest import game_from_prims
from . import bruteforce

EVENTS = [(s, o) for s in "+-" for o in ("rej", "succ", "fail")]
OC = {"rej": None, "succ": True, "fail": False}
PAPER = (F(1, 5), F(4, 5), F(3, 20), F(17, 20), F(3, 5))
STD, INV = {"+": 1, "-": 0}, {"+": 0, "-": 1}

# name, primitives (θ_L, θ_H, ρ⁻, ρ⁺, ρ*), h, μ, κ, σ, pinned event, α, user
INSTANCES = [
    ("SD3", PAPER, F(7, 8), F(1, 40), F(1, 9), (F(1), F(456, 16405)), "fail+", F(133, 153), STD),
    ("SD3'", PAPER, F(17, 20), F(1, 40), F(1, 9), (F(1), F(35, 579)), "rej-", F(47, 180), STD),
    ("SD6", PAPER, F(17, 20), F(1, 40), F(12, 13), (F(1), F(4012, 8685)), "succ+", F(20, 39), STD),
    ("BI1", PAPER, F(3, 10), F(7, 10), F(1, 3), (F(0), F(4078, 7021)), "fail-", F(31, 51), INV),
    ("BI2", PAPER, F(1, 40), F(3, 4), F(1, 9), (F(0), F(35, 507)), "rej+", F(47, 180), INV),
    (
        "BI3",
        (F(1, 2), F(9, 10), F(4, 5), F(19, 20), F(353, 400)),
        F(1, 20),
        F(1, 10),
        F(2, 5),
        (F(0), F(4211, 28880)),
        "succ-",
        F(3, 4),
        INV,
    ),
    ("SD3' outside tau_C", PAPER, F(1, 40), F(29, 40), F(1, 9), (F(1), F(7991, 8151)), "rej-", F(47, 180), STD),
    ("SD6 outside tau_C", PAPER, F(7, 10), F(1, 8), F(12, 13), (F(1), F(4862, 7713)), "succ+", F(20, 39), STD),
]


@pytest.mark.parametrize("inst", INSTANCES, ids=[i[0] for i in INSTANCES])
def test_instance_by_brute_force(inst) -> None:
    _, prims, h, mu, kappa, sig, pin, alpha, d = inst
    g = bruteforce.Game(*prims)
    ps = (g.rs - g.rm) / (g.rp - g.rs)
    delta = kappa / (1 + kappa)
    law = g.prior(h, mu)
    idx, posts = {}, {}
    for s, o in EVENTS:
        post = g.successor(law, sig, s == "+", OC[o])
        idx[o + s], posts[o + s] = (None if post is None else bruteforce.psi_hat(g, post)), post
    pinned = [k for k in idx if posts[k] == posts[pin]]
    assert idx[pin] == ps
    assert g.posterior_success(posts[pin], (F(1), F(1)), True) == g.rs  # terminal user exactly indifferent
    V = {k: alpha * g.c if k in pinned else (g.c if (v is None or v >= ps) else F(0)) for k, v in idx.items()}
    assert all(v is None or v != ps for k, v in idx.items() if k not in pinned)  # every other inequality strict
    D = g.gains(V, d, delta)
    rt = {s: g.posterior_success(law, sig, s == "+") for s in "+-"}
    assert all((d[s] == 1 and rt[s] >= g.rs) or (d[s] == 0 and rt[s] <= g.rs) for s in "+-")
    interior = 0 if 0 < sig[0] < 1 else 1
    assert D["-" if interior else "+"] == 0
    assert D["+"] > 0 if sig[0] == 1 else D["+"] < 0

    def N(x):  # the pinning function G − Ψ*·B on the unnormalized event law: affine in x
        sg = (sig[0], x) if interior else (x, sig[1])
        m = dict.fromkeys(law, F(0))
        for eta, ab, rho, hi, q in g.events(law, sg):
            if hi == (pin[-1] == "+"):
                m[(eta, ab)] += q * (1 if OC[pin[:-1]] is None else (rho if OC[pin[:-1]] else 1 - rho))
        G = sum(p * g.th[ab] for (_e, ab), p in m.items())
        B = sum(p * (1 - g.th[ab]) for (e, ab), p in m.items() if e == 0)
        return G - ps * B

    assert N(F(1)) != N(F(0))


@pytest.mark.parametrize("inst", INSTANCES, ids=[i[0] for i in INSTANCES])
def test_solver_finds_instance(inst) -> None:
    _, prims, h, mu, kappa, sig, pin, alpha, d = inst
    game = game_from_prims(*prims)
    found = [
        p
        for p in equilibria(game, h, mu, kappa / (1 + kappa))
        if p.sigma == sig and p.user == (d["+"], d["-"]) and p.terminal_mix is not None
    ]
    assert len(found) == 1
    events, a = found[0].terminal_mix
    assert pin in events.split("/") and a == alpha
    assert found[0].payoff_relevant
