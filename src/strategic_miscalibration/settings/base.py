"""What a setting provides to the runner: observations.

An :class:`Observation` is one row of data: the design columns that identify it, the model calls it
needs, and how to turn their answers into the row. The toy setting needs one call per observation; the
math Q&A setting needs two (the neutral instrument for ``ρ``, then the game). The runner does not know
the difference.

A new setting (a new task domain) is a module with an ``observations(config, ...)`` function returning
these; see ``docs/extending.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..llm.backends import Request
from ..llm.client import Answer

__all__ = ["Observation", "Row", "call_columns"]

Row = dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """One row to be produced.

    Attributes:
        key: Unique within a run; a resumed run skips keys already written.
        design: The columns fixed before any call (arm, state, draw, sample index, …).
        requests: The model calls, in order.
        finish: ``answers -> outcome columns``, given the answers in request order.
    """

    key: str
    design: Row
    requests: tuple[Request, ...]
    finish: Callable[[Sequence[Answer]], Row]

    def row(self, answers: Sequence[Answer]) -> Row:
        """The complete row."""
        return {**self.design, **self.finish(answers)}


def call_columns(answer: Answer, prefix: str = "") -> Row:
    """The provenance columns every call contributes: raw text, parse rung, failure, cost."""
    return {
        f"{prefix}raw": answer.raw,
        f"{prefix}rung": answer.rung,
        f"{prefix}failure": answer.failure,
        f"{prefix}cost_usd": answer.cost,
        f"{prefix}cached": answer.cached,
    }
