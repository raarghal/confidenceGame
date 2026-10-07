"""The statistics every result uses: paired comparisons over cells, intervals, slopes, variance shares.

Observations within a cell share a prompt and a state, so they are not independent: every interval
here resamples *cells* (or whatever the caller passes as the unit), never individual rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Paired", "bootstrap", "eta2", "ols", "paired_compare", "percentile_ci", "wilson"]

N_BOOT = 10_000
N_PERM = 10_000


@dataclass(frozen=True)
class Paired:
    """A paired comparison: ``arm − ref`` averaged over shared units, with a 95% interval and a p-value."""

    diff: float
    ci_lo: float
    ci_hi: float
    p_perm: float
    n: int
    ref_mean: float
    arm_mean: float

    @property
    def mde80(self) -> float:
        """Minimum detectable effect at 80% power, from the interval's implied standard error."""
        return 2.80 * (self.ci_hi - self.ci_lo) / (2 * 1.96)


def paired_compare(
    ref: pd.Series, arm: pd.Series, n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = 0
) -> Paired:
    """Cluster bootstrap and sign-flip permutation test on unit-level paired differences.

    ``ref`` and ``arm`` share an index of units (e.g. cells); units missing from either are dropped. The
    bootstrap resamples units, the same units for both sides, which keeps the comparison paired; under
    the null the sign of each paired difference is a fair coin.

    Raises:
        ValueError: If no unit is shared.
    """
    joined = pd.concat([ref.rename("ref"), arm.rename("arm")], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError("no shared units to compare")
    diffs = (joined["arm"] - joined["ref"]).to_numpy(float)
    n = len(diffs)
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    null = (rng.choice([-1.0, 1.0], size=(n_perm, n)) * diffs).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return Paired(obs, float(lo), float(hi), p, n, float(joined["ref"].mean()), float(joined["arm"].mean()))


def bootstrap(
    frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float | np.ndarray], n: int = 4000, seed: int = 0
) -> np.ndarray:
    """``statistic`` on ``n`` resamples of the rows of ``frame`` (one row per unit)."""
    rng = np.random.default_rng(seed)
    return np.array([statistic(frame.iloc[rng.integers(0, len(frame), len(frame))]) for _ in range(n)])


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """The central percentile interval of bootstrap draws (NaNs ignored)."""
    a = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(draws, [a, 100 - a])
    return float(lo), float(hi)


def ols(frame: pd.DataFrame, y: str, xs: Sequence[str]) -> np.ndarray:
    """Least-squares coefficients of ``y`` on ``xs`` (with an intercept, which is not returned)."""
    X = np.column_stack([np.ones(len(frame))] + [frame[x].to_numpy(float) for x in xs])
    coef: np.ndarray = np.linalg.lstsq(X, frame[y].to_numpy(float), rcond=None)[0][1:]
    return coef


def eta2(frame: pd.DataFrame, by: str, col: str) -> float:
    """Share of the variance of ``col`` explained by the groups of ``by``."""
    grand = frame[col].mean()
    total = ((frame[col] - grand) ** 2).sum()
    if total == 0:
        return float("nan")
    return float(sum(len(g) * (g[col].mean() - grand) ** 2 for _, g in frame.groupby(by)) / total)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n``."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)
