"""An independent brute-force enumerator of the two-period game, for cross-checks.

Deliberately shares **no code** with the package: every period-2 decision is found by enumerating the
joint distribution over ``(η, θ, ρ, s, z)`` and summing. No trust-index formula, no marginals, no
Bayes engine. A bug in the package therefore cannot hide behind the same bug here.
"""

from __future__ import annotations

import random
from fractions import Fraction as F


class Game:
    """The paper's binary game at primitives ``θ_L, θ_H, ρ⁻, ρ⁺, ρ*`` and fee ``c``."""

    def __init__(self, thL, thH, rm, rp, rs, c=F(1, 10)):
        self.th = {"L": F(thL), "H": F(thH)}
        self.rm, self.rp, self.rs, self.c = F(rm), F(rp), F(rs), F(c)

    def events(self, law, sig):
        """Yield ``(η, ability, ρ, high?, prob)`` for one period under rule ``sig = (σ⁺, σ⁻)``."""
        for (eta, ab), p in law.items():
            if p == 0:
                continue
            t = self.th[ab]
            for easy, pr in ((True, t), (False, 1 - t)):
                rho = self.rp if easy else self.rm
                ph = (F(1) if easy else F(0)) if eta == 1 else (sig[0] if easy else sig[1])
                for high, ps in ((True, ph), (False, 1 - ph)):
                    q = p * pr * ps
                    if q:
                        yield eta, ab, rho, high, q

    def posterior_success(self, law, sig, high):
        num = den = F(0)
        for _eta, _ab, rho, h, q in self.events(law, sig):
            if h == high:
                num += q * rho
                den += q
        return None if den == 0 else num / den

    def successor(self, law, sig, high, outcome):
        """Normalized law after a report and an outcome (``None`` = not delegated)."""
        m = dict.fromkeys(law, F(0))
        for eta, ab, rho, h, q in self.events(law, sig):
            if h != high:
                continue
            m[(eta, ab)] += q if outcome is None else q * (rho if outcome else 1 - rho)
        tot = sum(m.values())
        return None if tot == 0 else {k: v / tot for k, v in m.items()}

    def V2(self, law):
        """Terminal value under agent-optimal selection: the strategic type reports high on both draws."""
        ps = self.posterior_success(law, (F(1), F(1)), True)
        return self.c if (ps is not None and ps >= self.rs) else F(0)

    def prior(self, h, mu):
        h, mu = F(h), F(mu)
        return {(1, "H"): h * mu, (1, "L"): h * (1 - mu), (0, "H"): (1 - h) * mu, (0, "L"): (1 - h) * (1 - mu)}

    def values(self, h, mu, sig):
        law = self.prior(h, mu)
        V = {}
        for high, lab in ((True, "+"), (False, "-")):
            for ev, oc in (("rej", None), ("succ", True), ("fail", False)):
                post = self.successor(law, sig, high, oc)
                V[ev + lab] = None if post is None else self.V2(post)
        return V

    def gains(self, V, d, delta):
        out = {}
        for rho, lab in ((self.rp, "+"), (self.rm, "-")):

            def W(s, rho=rho):
                vd = rho * V["succ" + s] + (1 - rho) * V["fail" + s]
                return d[s] * (delta * self.c + (1 - delta) * vd) + (1 - d[s]) * (1 - delta) * V["rej" + s]

            out[lab] = W("+") - W("-")
        return out

    def is_eq(self, h, mu, sig, d, kappa):
        """``(both best-respond?, gains)``, or ``None`` if a terminal belief is unreached."""
        delta = F(kappa) / (1 + F(kappa))
        law = self.prior(h, mu)
        rt = {"+": self.posterior_success(law, sig, True), "-": self.posterior_success(law, sig, False)}
        ok_user = all(
            rt[s] is None
            or (d[s] == 1 and rt[s] >= self.rs)
            or (d[s] == 0 and rt[s] <= self.rs)
            or (0 < d[s] < 1 and rt[s] == self.rs)
            for s in "+-"
        )
        V = self.values(h, mu, sig)
        if None in V.values():
            return None
        D = self.gains(V, d, delta)
        ok_agent = all(
            [
                sig[0] == 0 or D["+"] >= 0,
                sig[0] == 1 or D["+"] <= 0,
                sig[1] == 0 or D["-"] >= 0,
                sig[1] == 1 or D["-"] <= 0,
            ]
        )
        return ok_user and ok_agent, D


def psi_hat(g: Game, law) -> F | None:
    G = sum(p * g.th[ab] for (_eta, ab), p in law.items())
    B = sum(p * (1 - g.th[ab]) for (eta, ab), p in law.items() if eta == 0)
    return None if B == 0 else G / B


def rand_prims(rng: random.Random):
    """Valid primitives ``0 < θ_L < θ_H < 1`` and ``0 < ρ⁻ < ρ* < ρ⁺ < 1``, in hundredths."""
    a, b = sorted(rng.sample(range(1, 100), 2))
    x, z = sorted(rng.sample(range(1, 100), 2))
    rm, rp = F(x, 100), F(z, 100)
    rs = rm + (rp - rm) * F(rng.randint(1, 99), 100)
    return F(a, 100), F(b, 100), rm, rp, rs
