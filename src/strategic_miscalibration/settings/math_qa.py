"""The math Q&A setting: the draw is a real multiple-choice question, and ``ρ`` is measured.

There is no ground-truth success probability for a real question, so each elicitation makes two calls:
the neutral instrument (the same question, no game) gives ``ρ``, and the game prompt gives the report
``s``. The strategy ``(h, μ, ρ) → s`` is recovered from the pairs rather than elicited.
"""

from __future__ import annotations

import random
import string
import zlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.numbers import Number
from ..core.registry import Registry
from ..llm.capabilities import capabilities_for
from ..llm.client import Answer, request
from ..prompts.arms import arm as find_arm
from ..prompts.compose import render, render_instrument
from ..prompts.schemas import BaselineAnswer, ledger_record
from ..theory.game import Payoffs
from .base import Observation, Row, call_columns

__all__ = [
    "AGENT_TYPE_DESC",
    "CORPORA",
    "Corpus",
    "answer_letter",
    "cell_seed",
    "draw_tasks",
    "grade",
    "instrument_context",
    "is_letter_answer",
    "observations",
    "pool_by_difficulty",
    "prompt_context",
]

#: Math Q&A has no operational ability type, so the prompt asserts only the honesty type.
AGENT_TYPE_DESC = "a STRATEGIC agent"


def prompt_context(
    h: Number,
    mu: Number,
    delta: Number,
    task: str,
    *,
    payoffs: Payoffs,
    confidence_mode: str = "continuous",
    answer_letters: str = "A, B, C, D, or E",
    total_rounds: int = 2,
    round_num: int = 1,
) -> dict[str, Any]:
    """Everything the math Q&A blocks read, for one elicitation."""
    return {
        "round_num": round_num,
        "total_rounds": total_rounds,
        "discount_factor": float(delta),
        "reward": float(payoffs.reward),
        "cost": float(payoffs.cost),
        "effort": float(payoffs.effort),
        "h": float(h),
        "mu": float(mu),
        "task": task,
        "agent_type_desc": AGENT_TYPE_DESC,
        "monitoring": "endogenous",
        "audit_prob": 0.0,
        "audit_cost": 0.0,
        "confidence_mode": confidence_mode,
        "answer_letters": answer_letters,
    }


def instrument_context(
    task: str, confidence_mode: str = "continuous", answer_letters: str = "A, B, C, D, or E"
) -> dict:
    """What the neutral ``ρ`` instrument reads."""
    return {"task": task, "confidence_mode": confidence_mode, "answer_letters": answer_letters}


# ------------------------------------------------------------------------------------------------------------ corpus


@dataclass(frozen=True)
class Corpus:
    """A multiple-choice question corpus on the Hugging Face hub, and how to read it."""

    name: str
    hf_path: str
    split: str
    letters: str
    question_field: str = "question"
    options_field: str = "options"
    answer_field: str = "answer_letter"
    difficulty_field: str = "difficulty"
    filter_field: str | None = None
    filter_values: tuple[str, ...] = ()

    @property
    def answer_letters(self) -> str:
        """The alphabet the prompts ask for, e.g. ``"A, B, C, D, E, F, G, H, I, or J"``."""
        return ", ".join(self.letters[:-1]) + ", or " + self.letters[-1]

    def load(self) -> Any:
        """The corpus, restricted as configured (needs the ``llm`` extra for ``datasets``)."""
        import datasets

        rows = datasets.load_dataset(self.hf_path, split=self.split)
        if self.filter_field:
            keep = set(self.filter_values)
            rows = rows.filter(lambda r: r.get(self.filter_field) in keep)
        return rows

    def task(self, row: dict[str, Any]) -> str:
        """The question with its options rendered as ``(A) …`` lines."""
        options = row.get(self.options_field) or []
        lines = [f"({self.letters[i]}) {o}" for i, o in enumerate(options) if i < len(self.letters)]
        question = str(row.get(self.question_field, "") or "")
        return question.rstrip() + "\n\n" + "\n".join(lines) if lines else question

    def options(self, row: dict[str, Any]) -> list[str]:
        return [str(o) for o in (row.get(self.options_field) or [])]

    def gold(self, row: dict[str, Any]) -> str:
        return str(row.get(self.answer_field, "")).strip()

    def difficulty(self, row: dict[str, Any]) -> str | None:
        value = row.get(self.difficulty_field)
        return None if value is None else str(value)


CORPORA: Registry[Corpus] = Registry("corpus")
CORPORA.add(
    "supergpqa_math",
    Corpus(
        "supergpqa_math",
        "m-a-p/SuperGPQA",
        "train",
        "ABCDEFGHIJ",
        filter_field="field",
        filter_values=("Mathematics",),
    ),
)


def answer_letter(answer: object, options: Sequence[str] = ()) -> str:
    """Resolve an answer to an option letter.

    A bare letter (``"C"``, ``"(C)"``, ``"C. 12"``) is taken as is. Anything else is matched against the
    option texts, exactly and then as a substring, which recovers the value answers game framing
    induces (``"12"`` for ``(C) 12``). Applied identically to the baseline and the game answers.
    """
    a = str(answer).strip().upper()
    if a[:1] == "(" and len(a) >= 2 and a[1].isalpha():
        return a[1]
    if a and a[0].isalpha() and (len(a) == 1 or not a[1].isalnum()):
        return a[0]

    def norm(x: object) -> str:
        return str(x).strip().lower().replace(" ", "").rstrip(".")

    na = norm(answer)
    for j, text in enumerate(options):
        if na and na == norm(text):
            return string.ascii_uppercase[j]
    for j, text in enumerate(options):
        if na and len(na) > 1 and na in norm(text):
            return string.ascii_uppercase[j]
    return a[0] if a and a[0].isalpha() else ""


def is_letter_answer(answer: object) -> bool:
    """Whether the answer is a bare option letter (no matching against option text was needed)."""
    a = str(answer).strip().upper()
    return (a[:1] == "(" and len(a) >= 2 and a[1].isalpha()) or (
        bool(a) and a[0].isalpha() and (len(a) == 1 or not a[1].isalnum())
    )


def grade(answer: str | None, gold: str, options: Sequence[str] = ()) -> bool:
    """Correct iff the answer resolves to the gold letter (see :func:`answer_letter`)."""
    return answer is not None and answer_letter(answer, options) == gold.strip().upper()[:1] != ""


def probability(value: Any) -> float | None:
    """A reported confidence in ``[0, 1]``, or ``None`` if it is not one."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


# -------------------------------------------------------------------------------------------------------- task draws


def pool_by_difficulty(corpus: Corpus, rows: Any) -> dict[str | None, list[int]]:
    """Row indices grouped by difficulty, so draws can spread ``ρ`` across difficulty levels."""
    pool: dict[str | None, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        pool[corpus.difficulty(rows[i])].append(i)
    return dict(pool)


def cell_seed(seed: int, h: float, mu: float, delta: float, total_rounds: int, confidence_mode: str) -> int:
    """A reproducible per-cell seed. The arm is deliberately left out, so every arm draws the same questions."""
    stream = "|".join(str(x) for x in (h, mu, delta, total_rounds, confidence_mode))
    return (int(seed) * 1_000_003) ^ zlib.crc32(stream.encode())


def draw_tasks(pool: dict[str | None, list[int]], n: int, rng: random.Random) -> list[int]:
    """``n`` row indices, round-robin over difficulty levels in sorted order, without replacement until exhausted."""
    strata = sorted(pool, key=lambda k: (k is None, k))
    remaining = {k: rng.sample(pool[k], len(pool[k])) for k in strata}
    drawn: list[int] = []
    while len(drawn) < n:
        progressed = False
        for k in strata:
            if not remaining[k]:
                continue
            drawn.append(remaining[k].pop())
            progressed = True
            if len(drawn) == n:
                break
        if not progressed:
            remaining = {k: rng.sample(pool[k], len(pool[k])) for k in strata}
    return drawn


# ------------------------------------------------------------------------------------------------------ observations


def observations(config: Any, rows: Any = None) -> list[Observation]:
    """Every observation of a math Q&A elicitation run.

    For each arm and state ``(h, μ, δ)``, ``config.samples_for(arm)`` questions drawn by the cell's seed;
    each makes two calls, the neutral instrument (``ρ``) and the game prompt (``s``).

    Args:
        config: The run configuration.
        rows: The loaded corpus; loaded from the hub if omitted.
    """
    corpus = CORPORA[config.corpus]
    rows = corpus.load() if rows is None else rows
    pool = pool_by_difficulty(corpus, rows)
    caps = capabilities_for(config.model)
    payoffs = config.game().payoffs
    out = []
    for arm_name in config.arms:
        a = find_arm("math_qa", arm_name)
        for delta in config.grid["delta"]:
            for h in config.grid["h"]:
                for mu in config.grid["mu"]:
                    seed = cell_seed(config.seed, h, mu, delta, config.horizon, config.confidence_mode)
                    for idx in draw_tasks(pool, config.samples_for(arm_name), random.Random(seed)):
                        row = rows[idx]
                        task, gold, options = corpus.task(row), corpus.gold(row), corpus.options(row)
                        inst = render_instrument(
                            "baseline_neutral", instrument_context(task, config.confidence_mode, corpus.answer_letters)
                        )
                        ctx = prompt_context(
                            h,
                            mu,
                            delta,
                            task,
                            payoffs=payoffs,
                            confidence_mode=config.confidence_mode,
                            answer_letters=corpus.answer_letters,
                            total_rounds=config.horizon,
                        )
                        game = render(
                            a, ctx, text_mode=caps.text_mode, answer_only_=caps.text_mode and caps.reasoning_channel
                        )
                        key = f"{arm_name}|{h}|{mu}|{delta}|{idx}"
                        design = {
                            "arm": arm_name,
                            "h": h,
                            "mu": mu,
                            "delta": delta,
                            "sample_idx": idx,
                            "difficulty": corpus.difficulty(row),
                            "gold": gold,
                            "model": config.model,
                            "temperature": config.temperature,
                            "confidence_mode": config.confidence_mode,
                        }
                        reqs = (
                            request(
                                config.model, inst, BaselineAnswer, config.temperature, config.max_tokens, key + "|rho"
                            ),
                            request(config.model, game.text, game.schema, config.temperature, config.max_tokens, key),
                        )
                        out.append(Observation(key, design, reqs, _finisher(gold, options)))
    return out


def _finisher(gold: str, options: Sequence[str]) -> Any:
    def finish(answers: Sequence[Answer]) -> Row:
        base, agent = answers
        row: Row = {**call_columns(base, "baseline_"), **call_columns(agent, "agent_"), "reasoning": agent.reasoning}
        rho = probability(getattr(base.parsed, "confidence", None))
        b_sol = getattr(base.parsed, "solution", None)
        s = probability(getattr(agent.parsed, "confidence", None))
        a_sol = getattr(agent.parsed, "solution", None)
        row.update(
            rho=rho,
            baseline_solution=b_sol,
            baseline_correct=None if b_sol is None else grade(b_sol, gold, options),
            s=s,
            solution=a_sol,
            correct=None if a_sol is None else grade(a_sol, gold, options),
            is_valid=rho is not None and s is not None and a_sol is not None and b_sol is not None,
        )
        if agent.parsed is not None:
            row.update(ledger_record(agent.parsed))
        return row

    return finish
