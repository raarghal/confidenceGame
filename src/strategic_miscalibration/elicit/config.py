"""A run, as a TOML file: which setting, which arms, which states, which model.

Example (``configs/toy/ledger.toml``)::

    name = "ledger_full"
    setting = "toy"
    model = "together_ai/openai/gpt-oss-120b"
    temperature = 0.7
    max_tokens = 4096
    arms = ["minimal_clarified_ledger"]
    samples = 20                      # per state and draw; or a table per arm: {minimal_clarified = 20}

    [grid]
    h = [0.1, 0.3, 0.5, 0.7, 0.9]
    mu = [0.1, 0.3, 0.5, 0.7, 0.9]
    delta = [0.05, 0.25, 0.45, 0.65]

    [game]                            # optional: the paper's game unless overridden
    audit_prob = 0.0                  # monitoring: 0 endogenous, 1 exogenous, in between an audit
    competitor = "L"                  # a transparent agent of this ability
    graded = {rho_plus_H = 0.95, rho_plus_L = 0.55}

The file is copied into every run directory, so it is the run's provenance.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..theory.game import Game, Monitoring, Roster, TransparentAgent, TypeSpace, paper_game

__all__ = ["RunConfig", "load"]


@dataclass(frozen=True)
class RunConfig:
    """One elicitation run. See the module docstring for the file format."""

    name: str
    setting: Literal["toy", "math_qa"]
    model: str
    temperature: float
    arms: tuple[str, ...]
    samples: int | dict[str, int]
    grid: dict[str, tuple[float, ...]]
    max_tokens: int | None = None
    conjecture: bool = False
    agent_ability: str = "H"
    horizon: int = 2
    seed: int = 12345
    corpus: str = "supergpqa_math"
    confidence_mode: str = "continuous"
    game_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.grid) != {"h", "mu", "delta"}:
            raise ValueError(f"{self.name}: [grid] needs exactly h, mu and delta")
        if isinstance(self.samples, dict) and set(self.samples) != set(self.arms):
            raise ValueError(f"{self.name}: samples must give a count for every arm")

    def samples_for(self, arm: str) -> int:
        return self.samples[arm] if isinstance(self.samples, dict) else self.samples

    def game(self) -> Game:
        """The paper's game with this run's overrides."""
        o = self.game_overrides
        game = paper_game()
        types = game.types
        if "graded" in o:
            g = o["graded"]
            types = TypeSpace.graded("0.8", "0.2", g["rho_plus_H"], g["rho_plus_L"], "0.15")
        kwargs: dict[str, Any] = {"types": types, "horizon": self.horizon}
        if "audit_prob" in o or "audit_cost" in o:
            kwargs["monitoring"] = Monitoring(o.get("audit_prob", 0), o.get("audit_cost", 0))
        if "competitor" in o:
            kwargs["roster"] = Roster((TransparentAgent("T", o["competitor"]),))
        return paper_game(**kwargs)

    @property
    def n_cells(self) -> int:
        return len(self.grid["h"]) * len(self.grid["mu"]) * len(self.grid["delta"])


def load(path: Path) -> RunConfig:
    """Read a run configuration."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    grid = {k: tuple(float(x) for x in v) for k, v in raw.pop("grid").items()}
    overrides = raw.pop("game", {})
    raw["arms"] = tuple(raw["arms"])
    return RunConfig(grid=grid, game_overrides=overrides, **raw)
