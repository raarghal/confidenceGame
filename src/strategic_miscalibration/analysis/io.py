"""Load a run: the one function every analysis reads data through.

A run directory (packed under ``data/runs/`` or written by :mod:`..elicit.runner`) holds ``rows.csv.gz``,
``config.toml`` and ``manifest.json``. :func:`load_run` returns them together, with the columns typed, so
no analysis re-implements "find the newest CSV".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.env import data_dir
from ..elicit.config import RunConfig, load

__all__ = ["Run", "load_run", "run_dir"]

_BOOL = ("is_valid", "conjecture", "report_high", "correct", "baseline_correct")


@dataclass(frozen=True)
class Run:
    """A run's rows, configuration and manifest."""

    name: str
    rows: pd.DataFrame
    config: RunConfig
    manifest: dict[str, Any]

    @property
    def valid(self) -> pd.DataFrame:
        """The rows with a usable answer."""
        return self.rows[self.rows["is_valid"]]

    def arm(self, name: str, conjecture: bool | None = None) -> pd.DataFrame:
        """Valid rows of one arm (optionally one conjecture setting)."""
        v = self.valid
        v = v[v["arm"] == name]
        return v if conjecture is None else v[v["conjecture"] == conjecture]


def run_dir(name: str) -> Path:
    """Where a packed run lives: ``$SMC_DATA_DIR/runs/<name>``."""
    return data_dir() / "runs" / name


@cache
def load_run(name_or_dir: str | Path) -> Run:
    """Load a run by name (from the data directory) or by path."""
    d = Path(name_or_dir) if Path(name_or_dir).is_dir() else run_dir(str(name_or_dir))
    if not (d / "rows.csv.gz").is_file():
        raise FileNotFoundError(f"no rows.csv.gz in {d}")
    rows = pd.read_csv(d / "rows.csv.gz", low_memory=False)
    for col in _BOOL:
        if col in rows:
            rows[col] = rows[col].map({True: True, False: False, "True": True, "False": False})
    rows["is_valid"] = rows["is_valid"].fillna(False).astype(bool)
    for col in ("h", "mu", "delta"):
        rows[col] = rows[col].astype(float).round(4)
    return Run(d.name, rows, load(d / "config.toml"), json.loads((d / "manifest.json").read_text()))
