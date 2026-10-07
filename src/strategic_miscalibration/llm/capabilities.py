"""How each model family must be talked to: the one table every model-conditional choice reads.

To add a model, add a row (most specific substring first). Measure before you trust: a provider that
*accepts* a parameter may ignore it, so check that the token spend actually moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

__all__ = ["CAPABILITIES", "DEFAULT", "Capabilities", "capabilities_for"]


@dataclass(frozen=True)
class Capabilities:
    """How to query one model family.

    Attributes:
        text_mode: Ask for plain text with a JSON skeleton in the prompt, and parse the answer out of it.
            Reasoning models do poorly under native schema enforcement. Otherwise the provider enforces
            the schema.
        reasoning_channel: The chain of thought arrives in a separate ``reasoning_content`` channel, so
            the requested JSON can omit ``reasoning`` and cannot be truncated by a long chain of thought.
        reasoning_effort: Value sent as ``reasoning_effort`` (a brake on chain-of-thought length), or
            ``None`` to send nothing.
        max_tokens: Default per-call budget.
        price_per_mtok: ``(input, output)`` USD per million tokens, for models the client library does
            not price itself.
        usd_per_call: Measured average cost of one call, for ``--dry-run`` estimates (``None`` if unmeasured).
        notes: How these values were established.
    """

    text_mode: bool = False
    reasoning_channel: bool = False
    reasoning_effort: str | None = None
    max_tokens: int = 512
    price_per_mtok: tuple[float, float] | None = None
    usd_per_call: float | None = None
    notes: str = ""


DEFAULT = Capabilities(notes="Instruction-tuned model with native JSON-schema support and no reasoning channel.")

CAPABILITIES: tuple[tuple[str, Capabilities], ...] = (
    (
        "gpt-oss",
        Capabilities(
            text_mode=True,
            reasoning_channel=True,
            reasoning_effort="low",
            max_tokens=2048,
            usd_per_call=0.0013,
            notes="The paper's model (gpt-oss-120b via Together). Without reasoning_effort='low' it can spend "
            "the whole budget reasoning and return no answer. usd_per_call: Together, Jul-Sep 2026 runs "
            "($0.0013 per toy call; the math Q&A runs averaged $0.00091 per two-call row).",
        ),
    ),
)


@cache
def capabilities_for(model: str) -> Capabilities:
    """The first row whose key is a substring of ``model`` (case-insensitive), else :data:`DEFAULT`."""
    lowered = model.lower()
    return next((caps for key, caps in CAPABILITIES if key in lowered), DEFAULT)
