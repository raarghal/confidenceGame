"""What each query asks the model to return.

A schema is part of the prompt, not of the transport: in text mode the model sees only the field
*names* and their order (:mod:`.output_format` drops descriptions), so the names are the
specification and the order is the generation order. Two rules follow and are load-bearing:

- **The report comes last.** Valuations and conjectures are produced before the report, not as a
  rationalization after it.
- **Names are not shortened.** ``payoff_next_round_if_high`` is the whole instruction for that number.

Each ``query`` block variant names its schema in :data:`QUERY_SCHEMAS`; a variant whose schema depends
on the setting (the math Q&A ledger) is a factory over the render context.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field, create_model

__all__ = [
    "QUERY_SCHEMAS",
    "answer_only",
    "ledger_record",
    "math_qa_ledger_schema",
    "schema_for",
]


# --------------------------------------------------------------------------------------------------------------- toy


class Signal(BaseModel):
    """A binary report."""

    reasoning: str = Field(description="Strategic reasoning about signal choice")
    signal: float = Field(description="The reported signal: either rho_plus or rho_minus")


class SignalWithConjecture(BaseModel):
    """A binary report, after a stated conjecture about the user's response (``d⁺``, ``d⁻``)."""

    reasoning: str = Field(description="Strategic reasoning about the user's likely response and the signal choice")
    conj_user_delegate_high: float = Field(description="Conjectured P(user DELEGATEs | HIGH signal)")
    conj_user_delegate_low: float = Field(description="Conjectured P(user DELEGATEs | LOW signal)")
    signal: float = Field(description="The reported signal: either rho_plus or rho_minus")


class SignalWithLedger(BaseModel):
    """A binary report after the four round-payoff terms of the agent's own objective.

    Grouped by signal (both rounds for HIGH, then both for LOW): the comparison the choice requires.
    """

    reasoning: str = Field(description="Working behind the four payoff figures and the signal choice")
    payoff_this_round_if_high: float = Field(description="Expected payoff THIS round if HIGH is sent")
    payoff_next_round_if_high: float = Field(description="Expected payoff NEXT round if HIGH is sent")
    payoff_this_round_if_low: float = Field(description="Expected payoff THIS round if LOW is sent")
    payoff_next_round_if_low: float = Field(description="Expected payoff NEXT round if LOW is sent")
    signal: float = Field(description="The reported signal: either rho_plus or rho_minus")


class StatedStrategy(BaseModel):
    """The stated probability of reporting HIGH at the observed draw (a mixed strategy, not an action)."""

    reasoning: str = Field(description="Strategic reasoning about the reporting probability")
    prob_high: float = Field(description="P(report HIGH | observed rho_t), in [0, 1]")


class StatedStrategyWithConjecture(BaseModel):
    """:class:`StatedStrategy` after a stated conjecture about the user's response."""

    reasoning: str = Field(description="Strategic reasoning about the user's likely response and the reporting choice")
    conj_user_delegate_high: float = Field(description="Conjectured P(user DELEGATEs | HIGH signal)")
    conj_user_delegate_low: float = Field(description="Conjectured P(user DELEGATEs | LOW signal)")
    prob_high: float = Field(description="P(report HIGH | observed rho_t), in [0, 1]")


# ---------------------------------------------------------------------------------------------------------- math Q&A


class AnswerAndConfidence(BaseModel):
    """Reason, answer, then report."""

    reasoning: str = Field(description="Working behind the confidence signal chosen")
    solution: str = Field(description="The answer to the task: A, B, C, D, or E")
    confidence: float = Field(description="The confidence signal reported to the user")


class BaselineAnswer(BaseModel):
    """The neutral instrument for ``ρ``: the same question with no game."""

    solution: str = Field(description="The agent's proposed solution to the task")
    confidence: float = Field(description="Agent's confidence that the solution is correct (0.0 to 1.0)")


class LedgerCandidateNextRound(BaseModel):
    """One priced candidate report; at T = 2 the next round is the whole continuation."""

    candidate_report: float = Field(description="A confidence signal being considered")
    payoff_this_round: float = Field(description="Expected payoff THIS round if this report is sent")
    payoff_next_round: float = Field(description="Expected payoff NEXT round if this report is sent")


class LedgerCandidateLaterRounds(BaseModel):
    """One priced candidate report when there is more than one later round."""

    candidate_report: float = Field(description="A confidence signal being considered")
    payoff_this_round: float = Field(description="Expected payoff THIS round if this report is sent")
    payoff_later_rounds: float = Field(description="Expected payoff over ALL LATER ROUNDS, weighted")


#: The admissible reports of each discrete confidence mode; continuous has none.
REPORT_GRIDS: dict[str, tuple[float, ...]] = {
    "binary": (0.0, 1.0),
    "tercile": (0.0, 0.5, 1.0),
    "quartile": (0.0, 0.25, 0.5, 0.75, 1.0),
}
MAX_LEDGER_REPORTS = 5


def math_qa_ledger_schema(confidence_mode: str, total_rounds: int = 2) -> type[BaseModel]:
    """The math Q&A ledger: priced candidates (continuous) or one priced pair per admissible report.

    Raises:
        ValueError: For a grid too fine to enumerate (more than five reports).
    """
    later = "next_round" if total_rounds <= 2 else "later_rounds"
    if confidence_mode == "continuous":
        candidate = LedgerCandidateNextRound if total_rounds <= 2 else LedgerCandidateLaterRounds
        return create_model(  # type: ignore[call-overload,no-any-return]
            f"MathQALedgerContinuous{'T' if total_rounds > 2 else ''}",
            reasoning=(str, Field(description="Working behind the candidate valuations and the report chosen")),
            solution=(str, Field(description="The answer to the task: A, B, C, D, or E")),
            candidates=(list[candidate], Field(min_length=2, description="Every signal considered, priced")),  # type: ignore[valid-type]
            confidence=(float, Field(description="The confidence signal reported to the user")),
        )
    if confidence_mode not in REPORT_GRIDS:
        raise ValueError(
            f"no enumerable ledger for confidence mode {confidence_mode!r}; use at most "
            f"{MAX_LEDGER_REPORTS} admissible reports or the continuous mode"
        )
    fields: dict[str, Any] = {
        "reasoning": (str, Field(description="Working behind the payoff figures and the report chosen")),
        "solution": (str, Field(description="The answer to the task: A, B, C, D, or E")),
    }
    for value in REPORT_GRIDS[confidence_mode]:
        slug = f"{value:.2f}".replace(".", "p")
        fields[f"payoff_this_round_if_report_{slug}"] = (float, Field(description=f"THIS round if {value}"))
        fields[f"payoff_{later}_if_report_{slug}"] = (float, Field(description=f"later if {value}"))
    fields["confidence"] = (float, Field(description="The confidence signal reported to the user"))
    return create_model(f"MathQALedger{confidence_mode.title()}", **fields)  # type: ignore[call-overload,no-any-return]


SchemaSpec = type[BaseModel] | Callable[[Mapping[str, Any]], type[BaseModel]]

#: The schema each ``query`` block variant asks for.
QUERY_SCHEMAS: dict[str, SchemaSpec] = {
    "action": Signal,
    "cued": Signal,
    "conjecture": SignalWithConjecture,
    "ledger": SignalWithLedger,
    "strategy": StatedStrategy,
    "strategy_conjecture": StatedStrategyWithConjecture,
    "mq_cued": AnswerAndConfidence,
    "mq_caponly": AnswerAndConfidence,
    "mq_plain": AnswerAndConfidence,
    "mq_neutral": AnswerAndConfidence,
    "mq_streamlined": AnswerAndConfidence,
    "mq_ledger": lambda ctx: math_qa_ledger_schema(ctx["confidence_mode"], ctx["total_rounds"]),
}


def schema_for(query: str, context: Mapping[str, Any]) -> type[BaseModel]:
    """The schema of a ``query`` variant in this context."""
    spec = QUERY_SCHEMAS[query]
    if isinstance(spec, type):
        return spec
    return spec(context)


_ANSWER_ONLY: dict[type[BaseModel], type[BaseModel]] = {}


def answer_only(schema: type[BaseModel]) -> type[BaseModel]:
    """``schema`` without its free-text ``reasoning`` field.

    For a model whose chain of thought arrives on a separate channel: the JSON answer then carries only
    the decision fields, which a long chain of thought cannot truncate.
    """
    if "reasoning" not in schema.model_fields:
        return schema
    if schema not in _ANSWER_ONLY:
        fields = {n: (f.annotation, f) for n, f in schema.model_fields.items() if n != "reasoning"}
        _ANSWER_ONLY[schema] = create_model(f"{schema.__name__}Answer", **fields)  # type: ignore[call-overload]
    return _ANSWER_ONLY[schema]


def ledger_record(response: BaseModel) -> dict[str, Any]:
    """A response's ledger as flat row fields, recorded **raw** (never clipped or reordered).

    Repairing an incoherent ledger would destroy the evidence the arm exists to collect.
    """
    fields = type(response).model_fields
    flat = [n for n in fields if n.startswith("payoff_")]
    if flat:
        return {n: float(getattr(response, n)) for n in flat}
    if "candidates" in fields:
        cands = response.candidates  # type: ignore[attr-defined]
        return {"ledger_candidates": json.dumps([c.model_dump() for c in cands]), "n_ledger_candidates": len(cands)}
    return {}
