"""Runs end to end, offline: call counts, the recorded question draws, the fake backend, resuming.

The fake corpus has the real corpus's row indices and difficulty levels (from ``fixtures/math_qa/pool.json``),
so the draws it produces are the draws the paper's runs made, with placeholder question text.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from strategic_miscalibration.elicit.config import RunConfig, load
from strategic_miscalibration.elicit.runner import plan, run
from strategic_miscalibration.llm.backends import FakeBackend, Request, fill_schema
from strategic_miscalibration.llm.client import Client
from strategic_miscalibration.settings.math_qa import cell_seed, draw_tasks

from ..conftest import FIXTURES

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
POOL = json.loads((FIXTURES / "math_qa" / "pool.json").read_text())
DRAWS = json.loads((FIXTURES / "math_qa" / "draws.json").read_text())

#: Rows each run wrote (the data on disk); a configuration must reproduce the count exactly.
RECORDED_ROWS = {
    "ledger_full": 4000,
    "c1_cued_full": 4000,
    "g3_clarified_full": 5000,
    "g3b_clarified_noconj": 4000,
    "e1_core": 5000,
    "wL_gate": 800,
    "e0_semantics_check": 800,
    "g1b_clarified_noconj": 800,
    "l0_link_power": 800,
    "g1_clarified_gate": 1000,
    "mathqa_map_supergpqa": 800,
    "mathqa_calib_length": 720,
    "mathqa_calib_decomp": 720,
    "mathqa_calib_hazard": 480,
}


def fake_corpus() -> list[dict]:
    """Rows with the real indices and difficulties; question text is a placeholder."""
    rows: list[dict] = [{} for _ in range(POOL["n_rows"])]
    for difficulty, idxs in POOL["pool"].items():
        for i in idxs:
            rows[i] = {
                "question": f"Question {i}?",
                "options": ["1", "2", "3"],
                "answer_letter": "B",
                "difficulty": difficulty,
            }
    return rows


def toy_answers(request: Request) -> dict:
    """A fake agent: a valid answer whose signal is ρ⁺ or ρ⁻ at random (seeded by the request)."""
    rng = random.Random(request.fingerprint())
    answer = fill_schema(request.schema, rng)
    if "signal" in answer:
        answer["signal"] = rng.choice([0.85, 0.15])
    return answer


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*/*.toml")), ids=lambda p: p.stem)
def test_call_counts_match_the_recorded_runs(path: Path) -> None:
    cfg = load(path)
    p = plan(cfg, fake_corpus() if cfg.setting == "math_qa" else None)
    assert p.observations == RECORDED_ROWS[cfg.name]
    assert p.calls == p.observations * (2 if cfg.setting == "math_qa" else 1)


def test_question_draws_reproduce_every_recorded_cell() -> None:
    pool = {k: v for k, v in POOL["pool"].items()}
    checked = 0
    for run_name, cells in DRAWS.items():
        if run_name.startswith("_"):
            continue
        for key, recorded in cells.items():
            _, h, mu, delta = key.split("|")
            seed = cell_seed(12345, float(h), float(mu), float(delta), 2, "continuous")
            assert draw_tasks(pool, len(recorded), random.Random(seed)) == recorded, (run_name, key)
            checked += 1
    assert checked > 100


def _small(cfg: RunConfig, **changes: object) -> RunConfig:
    grid = {"h": (0.3,), "mu": (0.7,), "delta": (0.05, 0.65)}
    return replace(
        cfg, grid=grid, samples=2 if isinstance(cfg.samples, int) else dict.fromkeys(cfg.arms, 2), **changes
    )


def test_toy_run_end_to_end_and_resume(tmp_path: Path) -> None:
    path = CONFIGS / "toy" / "ledger_full.toml"
    cfg = _small(load(path))
    backend = FakeBackend(toy_answers)
    table = run(cfg, path, tmp_path, Client(backend, max_workers=4))
    assert len(table) == 2 * 2 * 2 and table.is_valid.all()
    assert set(table.report_high) <= {True, False}
    assert {"payoff_this_round_if_high", "payoff_next_round_if_low", "rung", "raw"} <= set(table.columns)
    calls = backend.calls
    again = run(cfg, path, tmp_path, Client(backend))  # resuming a finished run makes no calls
    assert backend.calls == calls and len(again) == len(table)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rows"] == 8 and manifest["schema_version"] == 1
    assert pd.read_csv(tmp_path / "rows.csv.gz").shape[0] == 8


def test_toy_belief_elicited_run_records_conjectures(tmp_path: Path) -> None:
    path = CONFIGS / "toy" / "g3_clarified_full.toml"
    table = run(_small(load(path)), path, tmp_path, Client(FakeBackend(toy_answers)))
    action = table[table.arm == "minimal_clarified"]
    assert action.conj_delegate_high.between(0, 1).all()
    assert table[table.arm == "strategy_clarified"].prob_high.between(0, 1).all()


def test_math_qa_run_end_to_end(tmp_path: Path) -> None:
    path = CONFIGS / "math_qa" / "mathqa_map_supergpqa.toml"
    table = run(_small(load(path)), path, tmp_path, Client(FakeBackend()), corpus_rows=fake_corpus())
    assert len(table) == 2 * 2 * 2
    assert {"rho", "s", "baseline_correct", "correct", "ledger_candidates", "gold"} <= set(table.columns)
    assert table.gold.eq("B").all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"audit_prob": 0.5},
        {"audit_prob": 1.0},
        {"competitor": "L"},
        {"graded": {"rho_plus_H": 0.95, "rho_plus_L": 0.55}},
    ],
    ids=["audit", "exogenous", "competitor", "graded"],
)
def test_generalized_toy_games_run(tmp_path: Path, overrides: dict) -> None:
    """A game generalization is a config change: the same arms and runner, with the game's facts in the prompt."""
    path = CONFIGS / "toy" / "ledger_full.toml"
    cfg = _small(load(path), game_overrides=overrides)
    table = run(cfg, path, tmp_path, Client(FakeBackend(toy_answers)))
    assert table.is_valid.all()


def test_quantized_reports_run(tmp_path: Path) -> None:
    """A three-level confidence grid: the ledger enumerates one priced pair per admissible report."""
    path = CONFIGS / "math_qa" / "mathqa_map_supergpqa.toml"
    cfg = _small(load(path), confidence_mode="tercile")
    table = run(cfg, path, tmp_path, Client(FakeBackend()), corpus_rows=fake_corpus())
    ledger = table[table.arm == "minimal_clarified_ledger"]
    assert {"payoff_this_round_if_report_0p50", "payoff_next_round_if_report_1p00"} <= set(ledger.columns)
