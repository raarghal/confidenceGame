"""What does the measured strategy cost the user? (Section 6, Appendix P.)

Each state's measured rule is played against the exact two-period user of :mod:`..theory.welfare`. The
loss of a user is a share of the gains from trade, ``(W_honest − W_user) / (W_honest − W_0)``, where
``W_0 = 2(r − e)`` is never delegating. The naive user's loss splits exactly into *information destruction*
(``W_honest − W_soph``, lost to every user) and *exploitation* (``W_soph − W_naive``, recoverable by knowing
the rule).

Two accountings, as in the paper. In :func:`state_table` the population equals the user's belief, which
makes the naive and sophisticated users comparable. In :func:`capability_sweep` the population is fixed
apart from the belief: every agent strategic (``h* = 0``), a varying able share ``μ*``.

:func:`w4_strict_count` and :func:`wrong_population_witness` are the appendix's two exact statements about
knowing the rule; they use the theory alone, not the measured rules.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import numpy as np
import pandas as pd

from ..core.numbers import exact
from ..theory.beliefs import Strategy, binary_prior
from ..theory.game import HONEST, paper_game
from ..theory.welfare import benchmark, two_periods, user
from .behavior import PROTOCOLS, cell_rate, cells
from .io import load_run
from .stats import Paired, paired_compare

__all__ = [
    "capability_readings",
    "capability_sweep",
    "mutual_information",
    "nine_state_readings",
    "protocol_lever",
    "state_table",
    "summary",
    "type_independence",
    "w4_strict_count",
    "wrong_population_witness",
]

GAME = paper_game()
OUTSIDE = float(2 * GAME.payoffs.self_value())


def _rule(sp: float, sm: float) -> Strategy:
    return Strategy.binary(exact(float(sp)), exact(float(sm)))


def mutual_information(h: float, mu: float, sp: float, sm: float) -> float:
    """``I(draw; report)`` in bits, for the population ``(h, μ)`` playing the rule ``(σ⁺, σ⁻)``."""
    law = binary_prior(GAME, str(h), str(mu))
    joint = np.zeros((2, 2))
    for (eta, a), w in law.items():
        for i, draw in enumerate(("easy", "hard")):
            p_high = (1.0 if draw == "easy" else 0.0) if eta == HONEST else (sp if draw == "easy" else sm)
            mass = float(w) * float(GAME.types.p_draw(draw, a))
            joint[i] += mass * np.array([p_high, 1 - p_high])
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log2(joint / (px * py)), 0.0)
    return float(terms.sum())


@cache
def state_table(protocol: str = "ledger") -> pd.DataFrame:
    """Per state, with the population equal to the belief: honest benchmark, naive and sophisticated welfare,
    the loss and its split, and the informativeness of the report."""
    out = []
    for r in cells(PROTOCOLS[protocol].rows()).itertuples():
        state = binary_prior(GAME, str(r.h), str(r.mu))
        rule = _rule(r.sigma_plus, r.sigma_minus)
        naive = two_periods(GAME, user(GAME, "naive", state, state, rule), state, rule)
        soph = two_periods(GAME, user(GAME, "soph", state, state, rule), state, rule)
        bench = benchmark(GAME, state, state)
        out.append(
            {
                "h": r.h,
                "mu": r.mu,
                "delta": r.delta,
                "sigma_plus": r.sigma_plus,
                "sigma_minus": r.sigma_minus,
                "B": float(bench.user),
                "W_naive": float(naive.user),
                "W_soph": float(soph.user),
                "loss1": float(bench.user1 - naive.user1),
                "loss2": float(bench.user2 - naive.user2),
                "MI": mutual_information(r.h, r.mu, r.sigma_plus, r.sigma_minus),
            }
        )
    t = pd.DataFrame(out)
    t["gft"] = t.B - OUTSIDE
    t["total"] = t.B - t.W_naive
    t["destroyed"] = t.B - t.W_soph
    t["exploit"] = t.W_soph - t.W_naive
    t["naive_pct"] = 100 * t.total / t.gft
    t["soph_pct"] = 100 * t.destroyed / t.gft
    return t


def summary(protocol: str = "ledger") -> dict:
    """Section 6's headline numbers, pooled over the states (sums, not means of ratios)."""
    t = state_table(protocol)

    def share(g: pd.DataFrame, num: str, den: str) -> float:
        return float(100 * g[num].sum() / g[den].sum())

    return {
        "naive_loss_pct": share(t, "total", "gft"),
        "destruction_share_pct": share(t, "destroyed", "total"),
        "destruction_share_by_h": {h: share(g, "destroyed", "total") for h, g in t.groupby("h")},
        "naive_loss_by_delta": {d: share(g, "total", "gft") for d, g in t.groupby("delta")},
        "soph_below_outside": int((t.W_soph < OUTSIDE - 1e-12).sum()),
        "round2_share_pct": float(100 * t.loss2.sum() / (t.loss1.sum() + t.loss2.sum())),
    }


def capability_sweep(mus: np.ndarray | None = None, states: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two-period payoff of the three users against the true able share ``μ*`` at ``h* = 0``, averaged over states.

    Each state's measured rule is played at its own belief; only the true population moves.
    """
    mus = np.linspace(0, 1, 11) if mus is None else np.asarray(mus, float)
    states = cells(PROTOCOLS["ledger"].rows()) if states is None else states
    rows = []
    for m in mus:
        pop = binary_prior(GAME, 0, exact(float(m)))
        acc: dict[str, list[float]] = {"naive": [], "calib": [], "inf": [], "bench": []}
        for r in states.itertuples():
            state = binary_prior(GAME, str(r.h), str(r.mu))
            rule = _rule(r.sigma_plus, r.sigma_minus)
            for key, kind in (("naive", "naive"), ("calib", "soph"), ("inf", "informed")):
                acc[key].append(float(two_periods(GAME, user(GAME, kind, state, pop, rule), pop, rule).user))  # type: ignore[arg-type]
            acc["bench"].append(float(benchmark(GAME, state, pop).user))
        rows.append({"mu": float(m), **{k: float(np.mean(v)) for k, v in acc.items()}})
    t = pd.DataFrame(rows)
    t["outside"] = OUTSIDE
    return t


def _affine_crossing(y0: float, y1: float, level: float) -> float:
    """Where an affine function of ``μ*`` through ``(0, y0)``, ``(1, y1)`` reaches ``level``.

    NaN when the function is flat: it then never reaches ``level``, or sits on it for every ``μ*``. The second
    happens for the rule-knowing user's gain at states where it decides exactly as the naive user does.
    """
    if y1 == y0:
        return float("nan")
    return (level - y0) / (y1 - y0)


def capability_readings(states: pd.DataFrame | None = None) -> dict[str, float]:
    """At ``h* = 0``: the naive loss at ``μ* = 0`` and ``1``, its break-even, and where knowing the rule hurts.

    No user's decision depends on the population, so the naive and rule-knowing curves are affine in ``μ*`` and
    their crossings are exact.
    """
    t = capability_sweep(np.array([0.0, 1.0]), states)
    gft = t.bench - t.outside
    return {
        "naive_loss_pct_mu0": float(100 * (t.bench[0] - t.naive[0]) / gft[0]),
        "naive_loss_pct_mu1": float(100 * (t.bench[1] - t.naive[1]) / gft[1]),
        "breakeven_mu": _affine_crossing(t.naive[0], t.naive[1], OUTSIDE),
        "rule_knowing_below_naive_from": _affine_crossing(t.calib[0] - t.naive[0], t.calib[1] - t.naive[1], 0.0),
    }


def nine_state_readings() -> dict[str, tuple[float, float]]:
    """The break-even share and the ``μ* = 0`` loss computed separately at ``(h, μ) ∈ {0.1, 0.5, 0.9}²``."""
    states = cells(PROTOCOLS["ledger"].rows())
    readings = [
        capability_readings(g)
        for (h, mu), g in states.groupby(["h", "mu"])
        if h in (0.1, 0.5, 0.9) and mu in (0.1, 0.5, 0.9)
    ]
    be = [r["breakeven_mu"] for r in readings]
    loss = [r["naive_loss_pct_mu0"] for r in readings]
    return {"breakeven_mu": (min(be), max(be)), "naive_loss_pct_mu0": (min(loss), max(loss))}


def protocol_lever() -> pd.DataFrame:
    """Per protocol: the naive and sophisticated losses and the mean informativeness of the report."""
    out = []
    for name in PROTOCOLS:
        t = state_table(name)
        out.append(
            {
                "protocol": name,
                "MI": t.MI.mean(),
                "naive_pct": 100 * t.total.sum() / t.gft.sum(),
                "soph_pct": 100 * t.destroyed.sum() / t.gft.sum(),
            }
        )
    return pd.DataFrame(out).set_index("protocol").sort_values("MI")


def w4_strict_count() -> tuple[int, int]:
    """W4 (Appendix P): at how many (state, rule) pairs the sophisticated user strictly beats the naive one.

    Population equal to the belief, over states ``(h, μ) ∈ {0.1, …, 0.9}²`` and rules ``(σ⁺, σ⁻) ∈ {0, ¼, …, 1}²``,
    exact two-period payoffs.

    Returns:
        ``(strict, pairs)``.
    """
    grid = [Fraction(i, 10) for i in range(1, 10)]
    rules = [Strategy.binary(Fraction(a, 4), Fraction(b, 4)) for a in range(5) for b in range(5)]
    strict = pairs = 0
    for h in grid:
        for mu in grid:
            state = binary_prior(GAME, h, mu)
            for rule in rules:
                naive = two_periods(GAME, user(GAME, "naive", state, state, rule), state, rule).user
                soph = two_periods(GAME, user(GAME, "soph", state, state, rule), state, rule).user
                strict += soph > naive
                pairs += 1
    return strict, pairs


def wrong_population_witness() -> dict[str, Fraction]:
    """Remark P.4: with a wrong population belief, knowing the rule hurts.

    Belief ``(h, μ) = (½, ½)``, rule ``(1, ½)``, true population ``(h*, μ*) = (0, 1)``.

    Returns:
        The naive user's round-1 and round-2 payoffs, the sophisticated user's round-2 payoff, and the
        naive user's two-period advantage, in that order.
    """
    state, pop = binary_prior(GAME, Fraction(1, 2), Fraction(1, 2)), binary_prior(GAME, 0, 1)
    rule = Strategy.binary(1, Fraction(1, 2))
    naive = two_periods(GAME, user(GAME, "naive", state, pop, rule), pop, rule)
    soph = two_periods(GAME, user(GAME, "soph", state, pop, rule), pop, rule)
    return {"naive_r1": naive.user1, "naive_r2": naive.user2, "soph_r2": soph.user2, "gap": naive.user - soph.user}


def type_independence() -> dict:
    """Does a low-ability agent report like a high-ability one? (Appendix P, "Type independence".)

    ``wL_gate`` re-elicits at low ability on 20 states that are an exact subset of the belief-elicited run;
    every contrast is paired state for state. Also the naive user's two-period loss against an all-strategic,
    low-ability population under each rule.
    """
    low_rows = load_run("wL_gate").arm("minimal_clarified", True)
    high_all = load_run("g3_clarified_full").arm("minimal_clarified", True)
    high_rows = high_all[high_all.h.isin([0.1, 0.3]) & high_all.delta.isin([0.05, 0.65])]
    out: dict = {}
    for draw, key in (("HARD", "sigma_minus"), ("EASY", "sigma_plus")):
        out[key] = paired_compare(cell_rate(high_rows, draw), cell_rate(low_rows, draw))
    lo_cells, hi_cells = cell_rate(low_rows, "HARD"), cell_rate(high_rows, "HARD")
    out["states_lower"] = int((lo_cells < hi_cells.reindex(lo_cells.index)).sum())
    for label, rows in (("high", high_rows), ("low", low_rows)):
        s = cell_rate(rows, "HARD").rename("y").reset_index()
        h_slope: Paired = paired_compare(
            s[s.h == 0.1].set_index(["mu", "delta"]).y, s[s.h == 0.3].set_index(["mu", "delta"]).y
        )
        d_slope: Paired = paired_compare(
            s[s.delta == 0.05].set_index(["h", "mu"]).y, s[s.delta == 0.65].set_index(["h", "mu"]).y
        )
        out[f"{label}_h_slope"], out[f"{label}_delta_slope"] = h_slope.diff, d_slope.diff
        pop = binary_prior(GAME, 0, 0)
        loss = gft = Fraction(0)
        for r in cells(rows).itertuples():
            state = binary_prior(GAME, str(r.h), str(r.mu))
            rule = _rule(r.sigma_plus, r.sigma_minus)
            bench = benchmark(GAME, state, pop).user
            loss += bench - two_periods(GAME, user(GAME, "naive", state, pop, rule), pop, rule).user
            gft += bench - 2 * GAME.payoffs.self_value()
        n = len(cells(rows))
        out[f"{label}_rule_loss"] = float(loss / n)
        out[f"{label}_rule_loss_pct"] = float(100 * loss / gft)
    return out
