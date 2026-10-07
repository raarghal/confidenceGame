"""Does it carry over to real tasks? The math Q&A setting (Appendix O and Section 5's last paragraph).

Each row asks one SuperGPQA mathematics question twice: plainly (its confidence ``ρ`` is the agent's own
estimate of its chance of being right) and inside the game (its report ``s``). Both answers are graded
against the key, resolving value answers against the option texts (:func:`..settings.math_qa.grade`).
Rows sharing a question are not independent, so every interval resamples questions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import pandas as pd

from ..settings.math_qa import is_letter_answer
from .behavior import PROTOCOLS, cell_rate
from .io import load_run
from .stats import percentile_ci, wilson

__all__ = [
    "CALIBRATION_RUNS",
    "RUNS",
    "answer_format",
    "binarized_rule",
    "calibration",
    "levels",
    "pool",
    "prompt_variants",
]

RUNS = ("mathqa_map_supergpqa", "mathqa_calib_length", "mathqa_calib_decomp", "mathqa_calib_hazard")
CALIBRATION_RUNS = RUNS[1:]
CUED = "minimal_clarified_cued"
RHO_STAR = 0.6
#: Reliability bins, placed between the values the agent actually reports (it reports a few focal values).
BINS = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])


@cache
def pool(arm: str = CUED, runs: tuple[str, ...] = RUNS) -> pd.DataFrame:
    """Valid rows of one arm across runs, with ``base_ok`` and ``game_ok`` (the regraded correctness)."""
    frames = [load_run(r).arm(arm).assign(run=r) for r in runs if arm in set(load_run(r).rows.arm)]
    d = pd.concat(frames, ignore_index=True)
    d["base_ok"] = d.baseline_correct.eq(True)  # an answer that could not be graded is not correct
    d["game_ok"] = d.correct.eq(True)
    return d


def _clustered(
    d: pd.DataFrame, statistic: Callable[[dict[str, np.ndarray]], float], n: int, seed: int = 0
) -> tuple[float, float]:
    """Question-clustered bootstrap interval of ``statistic`` over the arrays ``rho, s, base_ok, game_ok``."""
    groups = [np.asarray(v) for v in d.groupby("sample_idx").indices.values()]
    arr = {c: d[c].to_numpy() for c in ("rho", "s", "base_ok", "game_ok")}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        out[i] = statistic({c: v[idx] for c, v in arr.items()})
    return percentile_ci(out)


def _mean_clustered(groups: list[np.ndarray], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    est = float(np.mean(np.concatenate(groups)))
    draws = [np.mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
    return (est, *percentile_ci(np.array(draws)))


def _ece(conf: np.ndarray, correct: np.ndarray) -> float:
    idx = np.digitize(conf, BINS[1:-1], right=True)
    total = 0.0
    for b in range(len(BINS) - 1):
        m = idx == b
        if m.any():
            total += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return float(total)


METRICS: dict[str, Callable[[dict[str, np.ndarray]], float]] = {
    "accuracy_change": lambda a: a["game_ok"].mean() - a["base_ok"].mean(),
    "confidence_change": lambda a: a["s"].mean() - a["rho"].mean(),
    "overconfidence_plain": lambda a: a["rho"].mean() - a["base_ok"].mean(),
    "overconfidence_game": lambda a: a["s"].mean() - a["game_ok"].mean(),
    "ece_change": lambda a: _ece(a["s"], a["game_ok"]) - _ece(a["rho"], a["base_ok"]),
    "discrimination_plain": lambda a: a["rho"][a["base_ok"]].mean() - a["rho"][~a["base_ok"]].mean(),
    "discrimination_game": lambda a: a["s"][a["game_ok"]].mean() - a["s"][~a["game_ok"]].mean(),
}


def calibration(n: int = 2000) -> dict[str, tuple[float, float, float]]:
    """Table 10: what the game framing does to calibration, each with a question-clustered interval."""
    d = pool()
    arr = {c: d[c].to_numpy() for c in ("rho", "s", "base_ok", "game_ok")}
    return {name: (float(f(arr)), *_clustered(d, f, n)) for name, f in METRICS.items()}


def levels() -> dict:
    """The pool, the stated confidence and accuracy, and the top of the scale."""
    d = pool()
    out: dict = {
        "rows": len(d),
        "questions": int(d.sample_idx.nunique()),
        "rows_by_run": d.groupby("run").size().to_dict(),
        "confidence": (d.rho.mean(), d.s.mean()),
        "accuracy": (d.base_ok.mean(), d.game_ok.mean()),
        "exactly_one": float((d.s == 1.0).mean()),
        "on_decile_grid": (
            float(np.isclose(d.rho * 10, np.round(d.rho * 10)).mean()),
            float(np.isclose(d.s * 10, np.round(d.s * 10)).mean()),
        ),
    }
    for col, ok, label in (("rho", "base_ok", "plain"), ("s", "game_ok", "game")):
        top = d[d[col] >= 0.9]
        out[f"top_{label}"] = {
            "rows": len(top),
            "share": len(top) / len(d),
            "stated": top[col].mean(),
            "accuracy": top[ok].mean(),
        }
    low, high = d[d.rho < RHO_STAR], d[d.rho >= RHO_STAR]
    out["below_rho_star"] = {
        "rows": len(low),
        "share": len(low) / len(d),
        "by_delta": low.groupby("delta").size().to_dict(),
        "confidence": low.rho.mean(),
        "accuracy": low.base_ok.mean(),
        "above_confidence": high.rho.mean(),
        "above_accuracy": high.base_ok.mean(),
    }
    return out


def answer_format() -> dict:
    """How often answers are not a bare option letter, and the accuracy change on letter-only rows."""
    d = pool().copy()
    d["plain_letter"] = d.baseline_solution.map(is_letter_answer)
    d["game_letter"] = d.solution.map(is_letter_answer)
    letters = d[d.plain_letter & d.game_letter]
    diff = letters.game_ok.to_numpy(float) - letters.base_ok.to_numpy(float)
    groups = [diff[np.asarray(v)] for v in letters.groupby("sample_idx").indices.values()]
    return {
        "non_letter_plain": 1 - d.plain_letter.mean(),
        "non_letter_game": 1 - d.game_letter.mean(),
        "letter_rows": len(letters),
        "letter_accuracy_change": _mean_clustered(groups),
        "letter_top_accuracy": (letters[letters.rho >= 0.9].base_ok.mean(), letters[letters.s >= 0.9].game_ok.mean()),
    }


def binarized_rule() -> dict:
    """The reporting rule binarized at ``ρ* = 0.6`` for both game arms, against the toy setting's same protocols.

    ``σ⁻`` is the share of questions the agent itself rates below ``ρ*`` that it reports at or above ``ρ*``;
    ``σ⁺`` the share of the rest it reports at or above ``ρ*``.
    """
    out: dict = {}
    for label, d in (("cued", pool()), ("ledger", pool("minimal_clarified_ledger").dropna(subset=["rho", "s"]))):
        for side, sub in (("-", d[d.rho < RHO_STAR]), ("+", d[d.rho >= RHO_STAR])):
            groups = [(g.s >= RHO_STAR).to_numpy(float) for _, g in sub.groupby("sample_idx")]
            k, n = int((sub.s >= RHO_STAR).sum()), len(sub)
            out[f"{label}{side}"] = {
                "rate": _mean_clustered(groups),
                "wilson": wilson(k, n),
                "rows": n,
                "questions": int(sub.sample_idx.nunique()),
            }
    for name in ("cued", "ledger"):
        rows = PROTOCOLS[name].rows()
        out[f"toy_{name}"] = (cell_rate(rows, "HARD").mean(), cell_rate(rows, "EASY").mean())
    return out


def prompt_variants() -> pd.DataFrame:
    """Accuracy change (in game minus plain) by prompt variant at ``h = μ = ½``, question-clustered."""
    arms = sorted({a for r in CALIBRATION_RUNS for a in load_run(r).rows.arm.unique()})
    out = []
    for arm in arms:
        d = pool(arm, CALIBRATION_RUNS)
        diff = d.game_ok.to_numpy(float) - d.base_ok.to_numpy(float)
        groups = [diff[np.asarray(v)] for v in d.groupby("sample_idx").indices.values()]
        out.append(
            {
                "arm": arm,
                "rows": len(d),
                "accuracy_change": _mean_clustered(groups),
                "non_letter": 1 - d.solution.map(is_letter_answer).mean(),
            }
        )
    return pd.DataFrame(out).set_index("arm")
