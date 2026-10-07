"""Is it what the theory predicts? And where it is not, where does the departure enter?

Everything here compares a measured strategy with the joint-law equilibrium of :mod:`..theory`, at the
experiments' primitives, state by state:

- **the regions** — each prior is in the robust region ``τR`` (trust survives a failed bluff), the fragile
  region ``τF`` (it survives a success only) or outside ``τ_C`` (no report is payoff-relevant), and the
  equilibrium bluffing rate there (:func:`state_table`);
- **the response to myopia** — the jump in ``σ⁻`` across the threshold ``δ* = 0.459`` by region, and
  whether the response below it is flat (:func:`myopia_response`);
- **the theory match by protocol** (:func:`protocol_match`, Table 8);
- **why it departs** — the agent's own objective and model of the user: its stated round-1 delegation
  belief ``d̂`` (:func:`belief_mediation`), the theory's decision rule with the agent's stated values
  substituted one at a time (:func:`substitution`), whether its report maximizes its own stated values,
  and what its model of the user costs it (:func:`forgone_payoff`, :func:`conjecture_accuracy`).

Intervals are bootstraps over states or over priors, as each docstring says; seeds are fixed so every
number is reproducible.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from ..theory.beliefs import Strategy, binary_prior, expected_success, posterior, weights
from ..theory.game import SELF, SUBJECT, paper_game
from ..theory.regions import predict
from ..theory.solvers import solve
from ..theory.terminal import is_trusted
from ..theory.users import OracleUser
from .behavior import PROTOCOLS, cells, derived_conjecture, ledger_coherence
from .io import load_run
from .stats import ols, percentile_ci

__all__ = [
    "SWEEP",
    "belief_mediation",
    "conjecture_accuracy",
    "family_ordering",
    "forgone_payoff",
    "glyph_table",
    "levels",
    "low_reports_on_easy_draws",
    "myopia_response",
    "protocol_match",
    "region_table",
    "state_table",
    "stated_continuation",
    "substitution",
    "under_reporting",
]

GAME = paper_game()
COST = float(GAME.cost)
DELTA_STAR = float(GAME.delta_star())
#: The sweep each comparative static is scaled to (h and μ over 0.1→0.9, δ over 0.05→0.65).
SWEEP = {"h": 0.8, "mu": 0.8, "delta": 0.6}


def _x(v: float) -> Fraction:
    """A measured rate as an exact number, rounded to 4 decimals (the convention the paper's numbers use)."""
    return Fraction(str(round(float(v), 4)))


# ------------------------------------------------------------------------------------------------------------ states


@cache
def _prediction(h: float, mu: float, delta: float) -> tuple[str, float]:
    p = predict(GAME, str(h), str(mu), str(delta))
    return p.region, float("nan") if p.sigma_minus is None else float(p.sigma_minus)


def true_user(h: float, mu: float, sigma_plus: float, sigma_minus: float) -> tuple[int, int]:
    """The round-1 Bayes user's response to the measured rule: delegate after HIGH, after LOW (1/0).

    Round 1 starts from the product prior, so this is the same on the joint law and on the marginals. A
    report the rule makes impossible is not delegated.
    """
    rule = Strategy.binary(_x(sigma_plus), _x(sigma_minus))
    law = binary_prior(GAME, str(h), str(mu))
    out = []
    for msg in ("HIGH", "LOW"):
        e = expected_success(GAME, law, rule, {SUBJECT: msg})
        out.append(int(e is not None and e >= GAME.rho_star))
    return out[0], out[1]


@cache
def state_table(protocol: str = "ledger") -> pd.DataFrame:
    """One row per state: measured ``σ±``, region, equilibrium ``σ⁻`` and the true user's response.

    For the ledger protocol, also the agent's derived delegation beliefs ``d̂(±)``, averaged over hard rows.
    """
    rows = PROTOCOLS[protocol].rows()
    c = cells(rows)
    pred = [_prediction(r.h, r.mu, r.delta) for r in c.itertuples()]
    c["region"] = [p[0] for p in pred]
    c["sigma_eq"] = [p[1] for p in pred]
    truth = [true_user(r.h, r.mu, r.sigma_plus, r.sigma_minus) for r in c.itertuples()]
    c["d_true_high"] = [t[0] for t in truth]
    c["d_true_low"] = [t[1] for t in truth]
    if protocol == "ledger":
        hard = derived_conjecture(rows[rows.draw == "HARD"], COST)
        d = hard.groupby(["h", "mu", "delta"])[["d_high", "d_low"]].mean().reset_index()
        c = c.merge(d, on=["h", "mu", "delta"])
    return c


def levels(protocol: str = "ledger") -> dict[str, Any]:
    """The severity numbers of Section 5: mean rates, spread, and how many states over-report."""
    c = state_table(protocol)
    sm, sp = c.sigma_minus, c.sigma_plus
    return {
        "sigma_minus": sm.mean(),
        "sigma_plus": sp.mean(),
        "sd_minus": sm.std(),
        "sd_plus": sp.std(),
        "max_minus": sm.max(),
        "states_ge_half": int((sm >= 0.5).sum()),
        "states_gt_half": int((sm > 0.5).sum()),
        "states": len(c),
        "trusted": int(c.region.isin(["R", "F", "D"]).sum()),
        "robust": int(c[c.delta == c.delta.min()].region.eq("R").sum()),
        "fragile": int(c[c.delta == c.delta.min()].region.eq("F").sum()),
        "restricted_level": c[(c.delta == c.delta.max()) & c.region.isin(["R", "F"])].sigma_minus.mean(),
    }


# ------------------------------------------------------------------------------------------------------------ myopia


def _per_prior(c: pd.DataFrame, col: str, f) -> pd.Series:
    return c.groupby(["h", "mu"]).apply(lambda g: f(g.set_index("delta")[col]), include_groups=False)


def _jump(s: pd.Series) -> float:
    """Rate above the threshold minus the mean of the rates below it."""
    return float(s[s.index > DELTA_STAR].mean() - s[s.index < DELTA_STAR].mean())


def myopia_response(protocol: str = "ledger", n: int = 10_000, seed: int = 20260925) -> dict[str, dict]:
    """The response to myopia by region, with cluster-bootstrap intervals over priors.

    Per prior: the *jump* (rate at δ = 0.65 minus the mean below the threshold), the change below the
    threshold (0.05 → 0.45), the *linearity shortfall* (a response linear in δ puts two thirds of the
    0.05 → 0.65 rise below 0.45; how far the measured change falls short of that), and the full rise.
    """
    c = state_table(protocol)
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for region in ("R", "F", "out"):
        g = c[c.region == region]
        stats = {
            "jump": _per_prior(g, "sigma_minus", _jump),
            "below": _per_prior(g, "sigma_minus", lambda s: s[0.45] - s[0.05]),
            "shortfall": _per_prior(g, "sigma_minus", lambda s: 2 / 3 * (s[0.65] - s[0.05]) - (s[0.45] - s[0.05])),
            "rise": _per_prior(g, "sigma_minus", lambda s: s[0.65] - s[0.05]),
        }
        k = len(stats["jump"])
        draws = [rng.integers(0, k, k) for _ in range(n)]
        out[region] = {"priors": k, "jumping_priors": int((stats["jump"] > 0).sum())}
        for name, s in stats.items():
            v = s.to_numpy()
            out[region][name] = (float(v.mean()), *percentile_ci(np.array([v[i].mean() for i in draws])))
    rf = c[c.region.isin(["R", "F"])]
    rise = _per_prior(rf, "sigma_minus", lambda s: s[0.65] - s[0.05]).to_numpy()
    out["R+F"] = {
        "rise": (
            float(rise.mean()),
            *percentile_ci(np.array([rise[rng.integers(0, len(rise), len(rise))].mean() for _ in range(n)])),
        )
    }
    return out


def region_table(protocol: str = "ledger") -> pd.DataFrame:
    """Table 7: mean ``σ⁻`` over priors by region and δ, with the equilibrium rate."""
    c = state_table(protocol)
    measured = c.groupby(["region", "delta"]).sigma_minus.mean().unstack()
    eq = c[c.region != "out"].groupby(["region", "delta"]).sigma_eq.mean().unstack()
    return pd.concat({"measured": measured, "equilibrium": eq})


def family_ordering(protocol: str = "ledger") -> dict[str, dict]:
    """The equilibrium family on ``τF`` below the threshold (SD3 or SD3′), and whether the measured rates order
    the priors as the equilibrium rates do (concordant pairs among the family's priors)."""
    c = state_table(protocol)
    f = c[(c.region == "F") & (c.delta < DELTA_STAR)]
    per = f.groupby(["h", "mu"]).agg(measured=("sigma_minus", "mean"), equilibrium=("sigma_eq", "first")).reset_index()
    per["family"] = [_fragile_family(r.h, r.mu) for r in per.itertuples()]
    out = {}
    for fam, g in per.groupby("family"):
        pairs = list(itertools.combinations(g.itertuples(), 2))
        agree = sum((a.measured - b.measured) * (a.equilibrium - b.equilibrium) > 0 for a, b in pairs)
        out[fam] = {
            "priors": len(g),
            "pairs": len(pairs),
            "concordant": int(agree),
            "measured_range": (g.measured.min(), g.measured.max()),
            "eq_range": (g.equilibrium.min(), g.equilibrium.max()),
        }
    return out


def _fragile_family(h: float, mu: float) -> str:
    """``SD3`` (the boundary belief is the post-failure one) or ``SD3′`` (the post-rejection low one)."""
    std = [p for p in solve(GAME, str(h), str(mu), "0.05") if p.payoff_relevant and p.user == (1, 0)]
    events = std[0].terminal_mix[0] if std and std[0].terminal_mix else ""
    return "SD3" if "fail+" in events else "SD3'" if "rej-" in events else "other"


def under_reporting(protocol: str = "ledger", n: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """The share of easy draws reported LOW, below and above ``h = ½``, and the gap with a bootstrap over priors."""
    rows = PROTOCOLS[protocol].rows()
    easy = rows[rows.draw == "EASY"].assign(low=lambda d: (~d.report_high.astype(bool)).astype(float))
    by_prior = easy.groupby(["h", "mu"]).low.agg(["sum", "count"]).reset_index()
    lo, hi = by_prior[by_prior.h < 0.5], by_prior[by_prior.h > 0.5]
    rng = np.random.default_rng(seed)

    def share(g: pd.DataFrame, i: np.ndarray) -> float:
        return float(g["sum"].to_numpy()[i].sum() / g["count"].to_numpy()[i].sum())

    gap = share(lo, np.arange(len(lo))) - share(hi, np.arange(len(hi)))
    draws = [
        share(lo, rng.integers(0, len(lo), len(lo))) - share(hi, rng.integers(0, len(hi), len(hi))) for _ in range(n)
    ]
    c = state_table(protocol)
    by_h = easy.groupby("h").low.mean()
    return {
        "below_half": share(lo, np.arange(len(lo))),
        "above_half": share(hi, np.arange(len(hi))),
        "gap": gap,
        "gap_ci": percentile_ci(np.array(draws)),
        "min_by_h": by_h.min(),
        "max_by_h": by_h.max(),
        "inverted_states": int(((c.sigma_plus < 1) & (c.sigma_minus > c.sigma_plus)).sum()),
    }


# -------------------------------------------------------------------------------------------------- across protocols

MATCH_ORDER = ("ledger", "cued", "semantics", "report-only", "belief-elicited")


def protocol_match(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Table 8: per protocol, the fit to the equilibrium rate over trusted states and the jumps.

    The difference of the jumps (``τF − τR``; positive in equilibrium) has a cluster bootstrap over priors.
    """
    rng = np.random.default_rng(seed)
    out = []
    for name in MATCH_ORDER:
        c = state_table(name)
        scored = c[c.region.isin(["R", "F", "D"]) & c.sigma_eq.notna()]

        def jump(cc: pd.DataFrame, region: str) -> float:
            s = cc[cc.region == region]
            below = s[s.delta < DELTA_STAR].groupby(["h", "mu"]).sigma_minus.mean()
            above = s[s.delta > DELTA_STAR].groupby(["h", "mu"]).sigma_minus.mean()
            return float((above - below).dropna().mean()) if len(s) else float("nan")

        priors = c[["h", "mu"]].drop_duplicates().to_numpy()
        draws = []
        for _ in range(n):
            pick = priors[rng.integers(0, len(priors), len(priors))]
            cc = pd.concat([c[(c.h == a) & (c.mu == b)] for a, b in pick])
            v = jump(cc, "F") - jump(cc, "R")
            if not np.isnan(v):
                draws.append(v)
        out.append(
            {
                "protocol": name,
                "sigma_minus": c.sigma_minus.mean(),
                "sigma_plus": c.sigma_plus.mean(),
                "r": float(np.corrcoef(scored.sigma_eq, scored.sigma_minus)[0, 1]),
                "mae": float((scored.sigma_eq - scored.sigma_minus).abs().mean()),
                "jump_F": jump(c, "F"),
                "jump_R": jump(c, "R"),
                "did": jump(c, "F") - jump(c, "R"),
                "did_ci": percentile_ci(np.array(draws)),
            }
        )
    return pd.DataFrame(out).set_index("protocol")


# ------------------------------------------------------------------------------------- the agent's model of the user


def belief_mediation(n: int = 4000, seed: int = 0) -> dict:
    """The departure runs through the agent's round-1 belief ``d̂(+)`` (ledger protocol; bootstrap over states).

    ``σ⁻`` regressed on ``(h, μ, δ)`` with and without ``d̂(+)`` (and with the true user's response instead),
    slopes scaled to each sweep; the correlation of ``d̂(+)`` with ``σ⁻``; and the same at δ = 0.65 inside ``τ_C``.
    """
    c = state_table("ledger")
    rng = np.random.default_rng(seed)
    sure = c[c.d_true_high == 1]
    base = ["h", "mu", "delta"]
    out: dict = {
        "states": len(c),
        "true_delegates_high": len(sure),
        "true_delegates_low": int(c.d_true_low.sum()),
        "dhat_high_where_true": sure.d_high.mean(),
        "dhat_low": c.d_low.mean(),
        "dhat_h_slope": ols(c, "d_high", ["h"])[0] * SWEEP["h"],
        "true_h_slope": ols(c, "d_true_high", ["h"])[0] * SWEEP["h"],
    }
    for label, cols in (("base", base), ("with_dhat", [*base, "d_high"]), ("with_true", [*base, "d_true_high"])):
        est = ols(c, "sigma_minus", cols)
        draws = np.array([ols(c.iloc[rng.integers(0, len(c), len(c))], "sigma_minus", cols) for _ in range(n)])
        res = {k: (est[i] * SWEEP[k], *percentile_ci(draws[:, i] * SWEEP[k])) for i, k in enumerate(base)}
        if len(cols) > 3:
            res[cols[-1]] = (est[-1], *percentile_ci(draws[:, -1]))
        out[label] = res

    def corr(s: pd.DataFrame) -> float:
        return float(np.corrcoef(s.d_high, s.sigma_minus)[0, 1])

    draws = np.array([corr(c.iloc[rng.integers(0, len(c), len(c))]) for _ in range(n)])
    out["corr"] = (corr(c), *percentile_ci(draws))
    top = c[(c.delta == 0.65) & c.region.isin(["R", "F", "D"])]
    est = ols(top, "sigma_minus", ["h", "mu"])
    d2 = np.array([ols(top.iloc[rng.integers(0, len(top), len(top))], "sigma_minus", ["h", "mu"]) for _ in range(n)])
    dc = np.array([corr(top.iloc[rng.integers(0, len(top), len(top))]) for _ in range(n)])
    out["restricted"] = {
        "states": len(top),
        "sigma_minus": top.sigma_minus.mean(),
        "h": (est[0] * 0.8, *percentile_ci(d2[:, 0] * 0.8)),
        "mu": (est[1] * 0.8, *percentile_ci(d2[:, 1] * 0.8)),
        "corr": (corr(top), *percentile_ci(dc)),
    }
    return out


def _continuation(h: float, mu: float, sp: float, sm: float) -> tuple[dict[str, float], dict[str, float]]:
    """The true user's round-1 response and the six continuation values (units of ``c``) under the measured rule."""
    rule = Strategy.binary(_x(sp), _x(sm))
    law = binary_prior(GAME, str(h), str(mu))
    d = {s: float(x) for s, x in zip("+-", true_user(h, mu, sp, sm), strict=True)}
    v = {}
    for s, msg in (("+", "HIGH"), ("-", "LOW")):
        for event, action, outcomes in (
            ("rej", SELF, {}),
            ("succ", SUBJECT, {SUBJECT: True}),
            ("fail", SUBJECT, {SUBJECT: False}),
        ):
            v[event + s] = float(is_trusted(GAME, weights(GAME, law, rule, {SUBJECT: msg}, action, outcomes)))
    return d, v


def _gain(delta: float, rho: float, d: dict[str, float], v: dict[str, float]) -> float:
    """``Δ(ρ)/c``: the gain from reporting HIGH rather than LOW, given delegation ``d`` and continuation ``v``."""

    def W(s: str) -> float:
        delegated = rho * v["succ" + s] + (1 - rho) * v["fail" + s]
        return d[s] * (delta + (1 - delta) * delegated) + (1 - d[s]) * (1 - delta) * v["rej" + s]

    return W("+") - W("-")


def _predicted(g: np.ndarray) -> np.ndarray:
    return np.where(g > 1e-12, 1.0, np.where(g < -1e-12, 0.0, 0.5))


@cache
def substitution_states() -> pd.DataFrame:
    """Per state, the high-report rate each model predicts (see :func:`substitution`)."""
    rows = PROTOCOLS["ledger"].rows()
    rp, rm = float(GAME.types.rho("easy", "H")), float(GAME.types.rho("hard", "H"))
    out = []
    for (h, mu, dl), g in rows.groupby(["h", "mu", "delta"]):
        e, hd = g[g.draw == "EASY"], g[g.draw == "HARD"]
        sp, sm = e.report_high.astype(float).mean(), hd.report_high.astype(float).mean()
        d_true, v_true = _continuation(h, mu, sp, sm)
        rec = {"h": h, "mu": mu, "delta": dl, "sp": sp, "sm": sm}
        for key, sub, rho in (("p", e, rp), ("m", hd, rm)):
            this_hi = (sub.payoff_this_round_if_high / COST).clip(0, 1).to_numpy()
            this_lo = (sub.payoff_this_round_if_low / COST).clip(0, 1).to_numpy()
            next_hi = (sub.payoff_next_round_if_high / COST).clip(0, 1).to_numpy()
            next_lo = (sub.payoff_next_round_if_low / COST).clip(0, 1).to_numpy()

            def own(a: float, b: float) -> dict[str, float]:
                return {"succ+": a, "fail+": a, "rej+": a, "succ-": b, "fail-": b, "rej-": b}

            rec[f"M1_{key}"] = _predicted(np.full(len(sub), _gain(dl, rho, d_true, v_true))).mean()
            rec[f"M2_{key}"] = _predicted(
                np.array([_gain(dl, rho, {"+": a, "-": b}, v_true) for a, b in zip(this_hi, this_lo, strict=True)])
            ).mean()
            rec[f"M2p_{key}"] = _predicted(
                np.array([_gain(dl, rho, d_true, own(a, b)) for a, b in zip(next_hi, next_lo, strict=True)])
            ).mean()
            V_hi = dl * sub.payoff_this_round_if_high + (1 - dl) * sub.payoff_next_round_if_high
            V_lo = dl * sub.payoff_this_round_if_low + (1 - dl) * sub.payoff_next_round_if_low
            rec[f"M3_{key}"] = _predicted((V_hi - V_lo).to_numpy()).mean()
        region, eq = _prediction(h, mu, dl)
        rec["region"] = region if region != "D" else "out"
        rec["M0_m"] = eq if region in ("R", "F") else np.nan
        out.append(rec)
    c = pd.DataFrame(out)
    c["const_m"] = c.sm.mean()
    return c


def _corr(frame: pd.DataFrame, col: str) -> float:
    return float(np.corrcoef(frame[col], frame.sm)[0, 1]) if frame[col].std() > 0 else float("nan")


def _slopes(c: pd.DataFrame, col: str) -> tuple[float, float, float]:
    b = ols(c, col, ["h", "mu", "delta"])
    return b[0] * SWEEP["h"], b[1] * SWEEP["mu"], b[2] * SWEEP["delta"]


def _region_jump(c: pd.DataFrame, col: str, region: str) -> float:
    s = c[c.region == region]
    return float(
        (s[s.delta > 0.5].groupby(["h", "mu"])[col].mean() - s[s.delta < 0.5].groupby(["h", "mu"])[col].mean()).mean()
    )


def substitution(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Table 9: the theory's decision rule with the agent's stated values substituted.

    M0 the equilibrium (trusted states); M1 the best response to the true game; M2 with the agent's
    round-1 belief ``d̂``; M2′ with its stated next-round values; M3 both (its own objective, unclipped).
    Each predicts a HIGH report iff ``Δ(ρ) > 0``. Scored by correlation with the measured ``σ⁻`` across
    states (bootstrap over states), slopes, jumps by region, and the error on ``σ⁺``.
    """
    c = substitution_states()
    rng = np.random.default_rng(seed)
    out = []
    for name, col in (("M0", "M0_m"), ("M1", "M1_m"), ("M2", "M2_m"), ("M2'", "M2p_m"), ("M3", "M3_m")):
        s = c[c[col].notna()]
        draws = np.array([_corr(s.iloc[rng.integers(0, len(s), len(s))], col) for _ in range(n)])
        pcol = col.replace("_m", "_p")
        out.append(
            {
                "model": name,
                "states": len(s),
                "r": float(np.corrcoef(s[col], s.sm)[0, 1]),
                "r_ci": percentile_ci(draws),
                "mae": float((s[col] - s.sm).abs().mean()),
                "slopes": _slopes(s, col),
                "jump_F": _region_jump(c, col, "F"),
                "jump_R": _region_jump(c, col, "R"),
                "mae_plus": float((c[pcol] - c.sp).abs().mean()) if pcol in c else np.nan,
            }
        )
    out.append(
        {
            "model": "measured",
            "states": len(c),
            "r": 1.0,
            "r_ci": (1.0, 1.0),
            "mae": 0.0,
            "slopes": _slopes(c, "sm"),
            "jump_F": _region_jump(c, "sm", "F"),
            "jump_R": _region_jump(c, "sm", "R"),
            "mae_plus": np.nan,
        }
    )
    out.append(
        {
            "model": "constant",
            "states": len(c),
            "r": np.nan,
            "r_ci": (np.nan, np.nan),
            "mae": float((c.const_m - c.sm).abs().mean()),
            "slopes": (0.0, 0.0, 0.0),
            "jump_F": 0.0,
            "jump_R": 0.0,
            "mae_plus": np.nan,
        }
    )
    return pd.DataFrame(out).set_index("model")


def model_jumps(n: int = 10_000, seed: int = 20260925) -> dict[str, dict]:
    """The jump on each region for the measured rule and for M2′, and their difference (bootstrap over priors).

    Also the share of hard-draw elicitations whose stated values price a bluff at ``≥ 0.25 c`` of next
    round's fee, by region.
    """
    c = substitution_states()
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for region in ("R", "F"):
        cc = c[c.region == region]
        jm = (
            cc[cc.delta > 0.5].groupby(["h", "mu"]).sm.mean() - cc[cc.delta < 0.5].groupby(["h", "mu"]).sm.mean()
        ).dropna()
        js = (
            cc[cc.delta > 0.5].groupby(["h", "mu"]).M2p_m.mean() - cc[cc.delta < 0.5].groupby(["h", "mu"]).M2p_m.mean()
        ).dropna()
        d = js - jm
        k = len(jm)
        idx = [rng.integers(0, k, k) for _ in range(n)]
        out[region] = {
            name: (float(v.mean()), *percentile_ci(np.array([v.to_numpy()[i].mean() for i in idx])))
            for name, v in (("measured", jm), ("M2'", js), ("difference", d))
        }
    return out


def stated_continuation(n: int = 10_000, seed: int = 1) -> pd.DataFrame:
    """The agent's stated next-round value after each report on hard draws (units of ``c``, clipped at the fee),
    by region, with the true value and a cluster bootstrap over priors; and the stated cost of a bluff where the
    agent reports a hard draw truthfully."""
    rows = PROTOCOLS["ledger"].rows()
    hard = rows[rows.draw == "HARD"].copy()
    hard["next_hi"] = (hard.payoff_next_round_if_high / COST).clip(0, 1)
    hard["next_lo"] = (hard.payoff_next_round_if_low / COST).clip(0, 1)
    hard["bluff_cost"] = hard.next_lo - hard.next_hi
    hard["region"] = [_prediction(h, mu, 0.05)[0] for h, mu in zip(hard.h, hard.mu, strict=True)]
    states = state_table("ledger").set_index(["h", "mu", "delta"])
    truth = {}
    for (h, mu, dl), r in states.iterrows():
        d, v = _continuation(h, mu, r.sigma_plus, r.sigma_minus)
        rho = float(GAME.types.rho("hard", "H"))
        truth[(h, mu, dl)] = {
            s: d[s] * (rho * v["succ" + s] + (1 - rho) * v["fail" + s]) + (1 - d[s]) * v["rej" + s] for s in "+-"
        }
    hard["true_hi"] = [truth[k]["+"] for k in zip(hard.h, hard.mu, hard.delta, strict=True)]
    hard["true_lo"] = [truth[k]["-"] for k in zip(hard.h, hard.mu, hard.delta, strict=True)]
    rng = np.random.default_rng(seed)
    out = []
    for region in ("R", "F"):
        g = hard[hard.region == region]
        truthful = g[~g.report_high.astype(bool)]
        rec: dict[str, Any] = {"region": region}
        for name, frame, col in (
            ("stated_hi", g, "next_hi"),
            ("stated_lo", g, "next_lo"),
            ("bluff_cost", truthful, "bluff_cost"),
        ):
            per = frame.groupby(["h", "mu"])[col].agg(["sum", "count"])
            k = len(per)
            draws = [
                per["sum"].to_numpy()[i].sum() / per["count"].to_numpy()[i].sum()
                for i in (rng.integers(0, k, k) for _ in range(n))
            ]
            rec[name] = (float(frame[col].mean()), *percentile_ci(np.array(draws)))
        rec["true_hi"], rec["true_lo"] = g.true_hi.mean(), g.true_lo.mean()
        rec["true_bluff_cost"] = (g.true_lo - g.true_hi).mean()
        out.append(rec)
    return pd.DataFrame(out).set_index("region")


def low_reports_on_easy_draws() -> dict[str, int]:
    """The easy draws the ledger agent reported LOW: do they maximize its stated values, and against the true
    user responding to the measured rule are they errors, payoff-irrelevant, or best responses?"""
    rows = PROTOCOLS["ledger"].rows()
    easy = rows[rows.draw == "EASY"]
    low = easy[~easy.report_high.astype(bool)]
    coh = ledger_coherence(low)
    states = state_table("ledger").set_index(["h", "mu", "delta"])
    rho = float(GAME.types.rho("easy", "H"))
    verdict = {"error": 0, "irrelevant": 0, "best_response": 0}
    at_high_trust = 0
    for r in low.itertuples():
        s = states.loc[(r.h, r.mu, r.delta)]
        d, v = _continuation(r.h, r.mu, s.sigma_plus, s.sigma_minus)
        g = _gain(r.delta, rho, d, v)
        kind = "error" if g > 1e-12 else "irrelevant" if abs(g) <= 1e-12 else "best_response"
        verdict[kind] += 1
        at_high_trust += kind == "best_response" and r.h >= 0.5
    via_next = int(((coh.payoff_next_round_if_low > coh.payoff_next_round_if_high) & coh.coherent).sum())
    return {
        "low_reports": len(low),
        "maximize_own_values": int(coh.coherent.sum()) + int(len(low) - len(coh)),
        "via_next_round": via_next,
        **verdict,
        "best_responses_at_h_ge_half": int(at_high_trust),
    }


def coherence() -> dict[str, Any]:
    """On the ledger protocol, the share of elicitations whose report maximizes ``δ v₁(s) + (1 − δ) v₂(s)``
    under the agent's own stated values, overall and by quartile of the stated margin (ties excluded)."""
    rows = PROTOCOLS["ledger"].rows()
    s = ledger_coherence(rows)
    s["q"] = pd.qcut(s.gap.abs(), 4, labels=False, duplicates="drop")
    return {
        "valid": len(rows),
        "ties": len(rows) - len(s),
        "scored": len(s),
        "share": s.coherent.mean(),
        "by_quartile": s.groupby("q").coherent.mean().round(3).tolist(),
    }


# ------------------------------------------------------------------------------------------ belief-elicited protocol


def _belief_elicited(kind: str = "action") -> pd.DataFrame:
    """Belief-elicited rows with the measured rule of their state and the true user's response to it."""
    run = load_run("g3_clarified_full")
    arm = "minimal_clarified" if kind == "action" else "strategy_clarified"
    rows = run.arm(arm, True).dropna(subset=["conj_delegate_high", "conj_delegate_low"]).copy()
    value = "report_high" if kind == "action" else "prob_high"
    c = cells(rows, value=value)
    truth = {
        (r.h, r.mu, r.delta): (r.sigma_plus, r.sigma_minus, *true_user(r.h, r.mu, r.sigma_plus, r.sigma_minus))
        for r in c.itertuples()
    }
    key = list(zip(rows.h, rows.mu, rows.delta, strict=True))
    rows["sp"], rows["sm"] = [truth[k][0] for k in key], [truth[k][1] for k in key]
    rows["oracle_high"], rows["oracle_low"] = [truth[k][2] for k in key], [truth[k][3] for k in key]
    return rows


def forgone_payoff() -> dict[str, float]:
    """Appendix N.1: payoff the belief-elicited agent forgoes against the true user, split exactly into "its
    model of the user is wrong" and "it does not act on its model" (joint-law continuation values)."""
    rows = _belief_elicited("action")
    rho = {"EASY": float(GAME.types.rho("easy", "H")), "HARD": float(GAME.types.rho("hard", "H"))}
    cache: dict = {}

    def node(r) -> dict[str, tuple[float, float]]:
        key = (r.h, r.mu, r.delta, r.draw)
        if key not in cache:
            rule = Strategy.binary(r.sp, r.sm)
            law = binary_prior(GAME, str(r.h), str(r.mu))
            out = {}
            for s, msg in (("+", "HIGH"), ("-", "LOW")):
                v = {}
                for event, action, outcomes in (
                    ("rej", SELF, {}),
                    ("succ", SUBJECT, {SUBJECT: True}),
                    ("fail", SUBJECT, {SUBJECT: False}),
                ):
                    post = posterior(GAME, law, rule, {SUBJECT: msg}, action, outcomes) or law
                    user = OracleUser(GAME, rule)
                    paid = any(user.decide(post, {SUBJECT: m}).action == SUBJECT for m in ("HIGH", "LOW"))
                    v[event] = COST if paid else 0.0
                p = rho[r.draw]
                out[s] = (p * v["succ"] + (1 - p) * v["fail"], v["rej"])
            cache[key] = out
        return cache[key]

    def U(r, d_hi: float, d_lo: float) -> tuple[float, float]:
        n = node(r)
        return tuple(  # type: ignore[return-value]
            d * (r.delta * COST + (1 - r.delta) * n[s][0]) + (1 - d) * (1 - r.delta) * n[s][1]
            for s, d in (("+", d_hi), ("-", d_lo))
        )

    model = act = best = 0.0
    for r in rows.itertuples():
        tru = U(r, r.oracle_high, r.oracle_low)
        own = U(r, r.conj_delegate_high, r.conj_delegate_low)
        u_star = max(tru)
        u_adv = tru[0] if own[0] > own[1] else tru[1]
        u_play = tru[0] if r.report_high else tru[1]
        model += u_star - u_adv
        act += u_adv - u_play
        best += u_star
    n = len(rows)
    forgone = (model + act) / n
    return {
        "n": n,
        "forgone_per_elicitation": forgone,
        "share_of_best": forgone / (best / n),
        "wrong_model": model / (model + act),
        "not_acting": act / (model + act),
    }


def conjecture_accuracy(band: float = 0.10, n: int = 8000, seed: int = 0) -> dict[str, dict]:
    """Hit rate of the stated conjecture (within ``band`` of the true user on both coordinates), against the best
    answer given at every state, with a bootstrap over states of the difference (action rows)."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("action", "strategy"):
        g = _belief_elicited(kind)
        oh, ol = g.oracle_high.to_numpy(float), g.oracle_low.to_numpy(float)
        hit = ((np.abs(g.conj_delegate_high - oh) <= band) & (np.abs(g.conj_delegate_low - ol) <= band)).astype(float)
        consts = [
            ((np.abs(a - oh) <= band) & (np.abs(b - ol) <= band)).astype(float) for a in (0.0, 1.0) for b in (0.0, 1.0)
        ]
        best = max(consts, key=lambda v: v.mean())
        per = pd.DataFrame({"c": list(zip(g.h, g.mu, g.delta, strict=True)), "a": hit.to_numpy(), "k": best})
        diff = (per.groupby("c").a.mean() - per.groupby("c").k.mean()).to_numpy()
        draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
        out[kind] = {
            "n": len(g),
            "accuracy": float(hit.mean()),
            "constant": float(best.mean()),
            "difference": (float(diff.mean()), *percentile_ci(draws)),
        }
    return out


def glyph_table(delta: float = 0.65) -> pd.DataFrame:
    """Figure 4: per belief state at one δ, the measured ``σ±``, the agent's derived ``d̂(±)`` (all rows), and
    the values the joint-law equilibrium correspondence admits for each.

    ``σ`` ticks come from payoff-relevant equilibria; ``d`` ticks from all equilibria. Where a total-rejection
    equilibrium exists the agent is never delegated and every ``σ`` is a best response (``sigma_free``).
    """
    rows = PROTOCOLS["ledger"].rows()
    rows = derived_conjecture(rows[rows.delta == delta], COST)
    c = cells(rows).merge(
        rows.groupby(["h", "mu", "delta"])[["d_high", "d_low"]].mean().reset_index(), on=["h", "mu", "delta"]
    )
    ticks, free = [], []
    for r in c.itertuples():
        eqs = solve(GAME, str(r.h), str(r.mu), str(r.delta))
        rel = [p for p in eqs if p.payoff_relevant]
        ticks.append(
            {
                "sigma_plus": sorted({round(float(p.sigma_plus), 3) for p in rel}),
                "sigma_minus": sorted({round(float(p.sigma_minus), 3) for p in rel}),
                "d_high": sorted({round(float(p.d_plus), 3) for p in eqs}),
                "d_low": sorted({round(float(p.d_minus), 3) for p in eqs}),
            }
        )
        free.append(any(not p.payoff_relevant and p.user == (0, 0) for p in eqs))
    c["ticks"], c["sigma_free"] = ticks, free
    return c
