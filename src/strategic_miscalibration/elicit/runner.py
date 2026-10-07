"""Run an elicitation: expand the grid, call the model in parallel, checkpoint, resume, record provenance.

A run directory holds::

    config.toml        the configuration, verbatim
    rows.jsonl         one row per observation, appended as batches finish (the checkpoint)
    rows.csv.gz        the finished table
    manifest.json      git revision, schema version, counts, failures, spend, timings

Rerunning the same configuration into the same directory skips every observation already in
``rows.jsonl``, so a run killed at any point loses at most one batch. With a cached backend the
skipped calls would also be free; resuming makes them unnecessary.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..llm.capabilities import capabilities_for
from ..llm.client import Client
from ..settings import math_qa, toy
from ..settings.base import Observation
from .config import RunConfig

__all__ = ["SCHEMA_VERSION", "Plan", "plan", "run"]

#: Bumped whenever a row column is added, removed or changes meaning.
SCHEMA_VERSION = 1
BATCH = 50


@dataclass(frozen=True)
class Plan:
    """What a run will do, before any call: for ``--dry-run``."""

    observations: int
    calls: int
    usd: float | None

    def __str__(self) -> str:
        cost = "unknown" if self.usd is None else f"${self.usd:,.2f}"
        return f"{self.observations:,} observations, {self.calls:,} calls, estimated cost {cost}"


def observations(config: RunConfig, corpus_rows: Any = None) -> list[Observation]:
    """Every observation the run makes, in a fixed order."""
    if config.setting == "toy":
        return toy.observations(config)
    return math_qa.observations(config, corpus_rows)


def plan(config: RunConfig, corpus_rows: Any = None) -> Plan:
    """Count observations, calls and cost without calling anything."""
    obs = observations(config, corpus_rows)
    calls = sum(len(o.requests) for o in obs)
    per_call = capabilities_for(config.model).usd_per_call
    return Plan(len(obs), calls, None if per_call is None else calls * per_call)


def run(config: RunConfig, config_path: Path, out_dir: Path, client: Client, corpus_rows: Any = None) -> pd.DataFrame:
    """Run (or resume) ``config`` into ``out_dir``; return the finished rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out_dir / "config.toml")
    checkpoint = out_dir / "rows.jsonl"
    done = _done_keys(checkpoint)
    todo = [o for o in observations(config, corpus_rows) if o.key not in done]
    started = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        answers = client.ask_all([r for o in batch for r in o.requests])
        rows, k = [], 0
        for o in batch:
            n = len(o.requests)
            rows.append({"key": o.key, **o.row(answers[k : k + n])})
            k += n
        with checkpoint.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, default=str) + "\n" for r in rows)
    table = pd.read_json(checkpoint, lines=True)
    with gzip.open(out_dir / "rows.csv.gz", "wt", encoding="utf-8") as fh:
        table.to_csv(fh, index=False)
    _write_manifest(out_dir, config, table, client, started, resumed=len(done))
    return table


def _done_keys(checkpoint: Path) -> set[str]:
    if not checkpoint.is_file():
        return set()
    with checkpoint.open(encoding="utf-8") as fh:
        return {json.loads(line)["key"] for line in fh if line.strip()}


def _write_manifest(
    out_dir: Path, config: RunConfig, table: pd.DataFrame, client: Client, started: float, resumed: int
) -> None:
    failures = table.filter(like="failure").apply(lambda c: c.notna().sum()).to_dict()
    manifest = {
        "name": config.name,
        "setting": config.setting,
        "schema_version": SCHEMA_VERSION,
        "rows": len(table),
        "valid": int(table["is_valid"].sum()),
        "failures": failures,
        "resumed_rows": resumed,
        "spend_usd_this_session": round(client.spent, 6),
        "seconds_this_session": round(time.time() - started, 1),
        "git": _git(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _git() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}
