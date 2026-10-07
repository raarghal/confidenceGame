"""What does the agent do? The measured strategy, and how it depends on the elicitation protocol.

The strategy, per state, is ``σ⁺ = P(report HIGH | easy)`` and ``σ⁻ = P(report HIGH | hard)``. A *cell* is a
state ``(h, μ, δ)``. Every other analysis (the theory fit, welfare) starts from :func:`cells`.

The ledger protocol's stated payoffs give two more
per-row objects: the round-1 delegation probability the agent prices, ``d̂(s) = payoff_this_round_if_s / c``
(:func:`derived_conjecture`; inferred, never asked), and whether the report maximizes the agent's own
stated values (:func:`ledger_coherence`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .io import load_run
from .stats import Paired, paired_compare

__all__ = [
    "CELL",
    "CHANNELS",
    "LEDGER",
    "PROTOCOLS",
    "Protocol",
    "cell_rate",
    "cells",
    "channel_slopes",
    "clause_effect",
    "derived_conjecture",
    "ledger_coherence",
    "misreads_counterpart",
    "over_reporting_states",
    "protocol_did",
    "protocol_slopes",
    "screened",
    "slope",
]

CELL = ["h", "mu", "delta"]
LEDGER = (
    "payoff_this_round_if_high",
    "payoff_next_round_if_high",
    "payoff_this_round_if_low",
    "payoff_next_round_if_low",
)


def cell_rate(rows: pd.DataFrame, draw: str, value: str = "report_high") -> pd.Series:
    """Per-cell mean of ``value`` on one draw (``"EASY"`` gives σ⁺, ``"HARD"`` gives σ⁻)."""
    sub = rows[rows["draw"] == draw]
    return sub.groupby(CELL)[value].apply(lambda s: s.astype(float).mean())


def cells(rows: pd.DataFrame, min_rows: int = 0, value: str = "report_high") -> pd.DataFrame:
    """One row per cell: ``sigma_plus``, ``sigma_minus`` and their row counts.

    Args:
        rows: Valid rows of one arm.
        min_rows: Drop cells with fewer rows than this on either draw.
        value: ``report_high`` for reports; ``prob_high`` for stated strategies.
    """
    out = pd.DataFrame(
        {
            "sigma_plus": cell_rate(rows, "EASY", value),
            "sigma_minus": cell_rate(rows, "HARD", value),
            "n_plus": rows[rows.draw == "EASY"].groupby(CELL).size(),
            "n_minus": rows[rows.draw == "HARD"].groupby(CELL).size(),
        }
    ).reset_index()
    keep = (out.n_plus >= min_rows) & (out.n_minus >= min_rows)
    return out[keep].reset_index(drop=True)


def slope(rates: pd.Series, coord: str, hold: Sequence[str], lo: float, hi: float) -> pd.Series:
    """Paired change along one coordinate: ``rate(hi) − rate(lo)`` for each setting of the others."""
    s = rates.rename("y").reset_index()
    a = s[s[coord] == lo].set_index(list(hold))["y"]
    b = s[s[coord] == hi].set_index(list(hold))["y"]
    return (b - a).dropna()


def derived_conjecture(rows: pd.DataFrame, cost: float) -> pd.DataFrame:
    """``d̂(+), d̂(−)``: the round-1 delegation probabilities the ledger agent prices, clipped to ``[0, 1]``.

    The prompt fixes the round-1 payoff at the fee ``c`` if delegated and 0 otherwise, so the stated
    payoff divided by ``c`` is the delegation probability the agent is pricing. It is an inference from
    stated payoffs, not an elicited belief.
    """
    out = rows.copy()
    out["d_high"] = (out["payoff_this_round_if_high"].astype(float) / cost).clip(0, 1)
    out["d_low"] = (out["payoff_this_round_if_low"].astype(float) / cost).clip(0, 1)
    return out


def ledger_coherence(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows where the stated values differ, each marked by whether the report maximizes them.

    The agent's own objective is ``V(s) = δ·(this round) + (1 − δ)·(next round)`` from its stated
    ledger. Ties (``|V(HIGH) − V(LOW)| ≤ 1e-9``) carry no information and are dropped.
    """
    d = rows[rows[list(LEDGER)].notna().all(axis=1)].copy()
    d["V_high"] = d.delta * d.payoff_this_round_if_high + (1 - d.delta) * d.payoff_next_round_if_high
    d["V_low"] = d.delta * d.payoff_this_round_if_low + (1 - d.delta) * d.payoff_next_round_if_low
    d["gap"] = d.V_high - d.V_low
    scored = d[d.gap.abs() > 1e-9].copy()
    scored["coherent"] = (scored.gap > 0) == scored.report_high.astype(bool)
    return scored


# --------------------------------------------------------------------------------------------------------- protocols


@dataclass(frozen=True)
class Protocol:
    """An elicitation protocol: the run and arm that measured it."""

    name: str
    run: str
    arm: str
    conjecture: bool

    def rows(self) -> pd.DataFrame:
        """Valid rows of this protocol's arm."""
        return load_run(self.run).arm(self.arm, self.conjecture)


#: The five protocols of Table 5 on the identical 100-state grid, in order of what the prompt supplies.
PROTOCOLS: dict[str, Protocol] = {
    p.name: p
    for p in (
        Protocol("semantics", "e1_core", "minimal_semantics", False),
        Protocol("report-only", "g3b_clarified_noconj", "minimal_clarified", False),
        Protocol("belief-elicited", "g3_clarified_full", "minimal_clarified", True),
        Protocol("cued", "c1_cued_full", "minimal_clarified_cued", False),
        Protocol("ledger", "ledger_full", "minimal_clarified_ledger", False),
    )
}

#: The three comparative statics: the coordinate swept, the coordinates held fixed, and the endpoints.
CHANNELS = {
    "delta": ("delta", ("h", "mu"), 0.05, 0.65),
    "h": ("h", ("mu", "delta"), 0.1, 0.9),
    "mu": ("mu", ("h", "delta"), 0.1, 0.9),
}


def channel_slopes(rows: pd.DataFrame, draw: str = "HARD") -> dict[str, pd.Series]:
    """Per-pair changes in the rate along each channel (the unit of every slope test)."""
    rates = cell_rate(rows, draw)
    return {name: slope(rates, coord, hold, lo, hi) for name, (coord, hold, lo, hi) in CHANNELS.items()}


def protocol_slopes(name: str) -> dict[str, Paired]:
    """Table 6: the over-reporting slope of one protocol along each channel, tested against zero."""
    return {
        ch: paired_compare(pd.Series(0.0, index=s.index), s)
        for ch, s in channel_slopes(PROTOCOLS[name].rows()).items()
    }


def protocol_did(ref: str, arm: str, channel: str) -> Paired:
    """Difference in one channel's slope between two protocols, paired over the held-fixed states."""
    return paired_compare(
        channel_slopes(PROTOCOLS[ref].rows())[channel], channel_slopes(PROTOCOLS[arm].rows())[channel]
    )


def over_reporting_states(name: str, threshold: float = 0.5) -> int:
    """Number of the 100 states at which ``σ⁻ ≥ threshold``."""
    return int((cell_rate(PROTOCOLS[name].rows(), "HARD") >= threshold).sum())


#: The matched 20-state clause ladder (conjecture off unless stated): the three runs on the gate grid.
CLAUSE_RUNS = {
    "semantics": ("e0_semantics_check", "minimal_semantics", False),
    "clarified": ("g1b_clarified_noconj", "minimal_clarified", False),
    "link": ("l0_link_power", "minimal_clarified_link", False),
    "clarified+conjecture": ("g1_clarified_gate", "minimal_clarified", True),
}


def clause_effect(ref: str, arm: str) -> Paired:
    """The change in ``σ⁻`` from one rung of the clause ladder to another, paired over the 20 matched states."""
    rates = {k: cell_rate(load_run(r).arm(a, c), "HARD") for k, (r, a, c) in CLAUSE_RUNS.items() if k in (ref, arm)}
    return paired_compare(rates[ref], rates[arm])


# --------------------------------------------------------------------------------------------- the misreading screen

_OPPOSITE = re.compile(r"\bopposite\b|\binvert", re.I)
_ABOUT_ABILITY = re.compile(r"\b(ability|able|theta|competen)", re.I)
_ABOUT_HONESTY = re.compile(r"\b(dishonest|honest|strategic|lie|lying|truthful|report|signal|send|claim)", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def misreads_counterpart(reasoning: str) -> bool:
    """Whether a trace assumes a dishonest agent reports the *opposite* of the truth.

    Under that misreading a low report looks trustworthy at low ``h``, so the agent under-reports in order
    to be delegated, a strictly dominated choice. The screen fires on "opposite"/"invert" in a sentence
    about honesty or reporting, and not when every such sentence is about the (genuinely mirrored)
    ability levels.
    """
    hits = [s for s in _SENTENCE.split(str(reasoning or "")) if _OPPOSITE.search(s)]
    if not hits or not any(_ABOUT_HONESTY.search(s) for s in hits):
        return False
    return not all(_ABOUT_ABILITY.search(s) and not _ABOUT_HONESTY.search(s) for s in hits)


def screened(rows: pd.DataFrame) -> pd.Series:
    """Per-row flag of :func:`misreads_counterpart`."""
    return rows["reasoning"].map(misreads_counterpart)
